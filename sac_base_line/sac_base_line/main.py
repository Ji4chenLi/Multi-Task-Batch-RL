import os
import os.path as osp

import pickle
import argparse
import torch
import importlib
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import utils.pytorch_util as ptu
from utils.pythonplusplus import load_gzip_pickle

from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector, tiMeSampler
from trainer.policies import TanhGaussianPolicy, MakeDeterministic
from trainer.trainer import SACTrainer
from networks import FlattenMlp, MlpEncoder, PerturbationGenerator, VaeDecoder
from rl_algorithm import BatchRLAlgorithm

from prob_context_encoder import ProbabilisticContextEncoder

import socket

import ray
import logging
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.WARNING,
    webui_host='127.0.0.1',

    # The amount of memory (in bytes)
    object_store_memory=1073741824 * 10, # 1g
    redis_max_memory=1073741824 * 10 # 1g
)


def get_current_branch(dir):

    from git import Repo

    repo = Repo(dir)
    return repo.active_branch.name


def get_policy_producer(obs_dim, action_dim, hidden_sizes):

    def policy_producer(deterministic=False):

        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

        if deterministic:
            policy = MakeDeterministic(policy)

        return policy

    return policy_producer


def get_q_producer(obs_dim, action_dim, hidden_sizes):
    def q_producer():
        return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=1,
                          hidden_sizes=hidden_sizes, )

    return q_producer


def experiment(variant, prev_exp_state=None):

    domain = variant['domain']
    seed = variant['seed']
    goal = variant['goal']

    expl_env = env_producer(domain, seed, goal)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    print('------------------------------------------------')
    print('obs_dim', obs_dim)
    print('action_dim', action_dim)
    print('------------------------------------------------')

    qf1 = FlattenMlp(
        hidden_sizes=variant['Qs_hidden_sizes'],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    target_qf1 = FlattenMlp(
        hidden_sizes=variant['Qs_hidden_sizes'],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=variant['Qs_hidden_sizes'],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    target_qf2 = FlattenMlp(
        hidden_sizes=variant['Qs_hidden_sizes'],
        input_size=obs_dim + action_dim,
        output_size=1,
    )

    # Get producer function for policy
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=variant['policy_hidden_sizes'])
    # Finished getting producer

    remote_eval_path_collector = RemoteMdpPathCollector.remote(
        domain, seed * 10 + 1,
        goal, policy_producer
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
    )
    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )
    trainer = SACTrainer(
        policy_producer,
        qf1=qf1,
        target_qf1=target_qf1,
        qf2=qf2,
        target_qf2=target_qf2,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )

    algorithm = BatchRLAlgorithm(
        trainer=trainer,

        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,

        replay_buffer=replay_buffer,
        optimistic_exp_hp=variant['optimistic_exp'],
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--goal', type=int, default=0)
    parser.add_argument('--config', type=str, default='ant-dir')
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)

    args = parser.parse_args()

    return args


def get_log_dir(variant, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    def create_simple_exp_name():
        """
        Create a unique experiment name with a timestamp
        """
        import datetime
        import dateutil

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        return timestamp

    log_dir = f'{task}_task_results'
    
    if should_include_domain:
        log_dir = osp.join(log_dir, variant['domain'])

    if should_include_seed:
        log_dir = osp.join(log_dir, f"seed_{variant['seed']}")

    log_dir = osp.join(log_dir, f"max_path_length_{variant['algorithm_kwargs']['max_path_length']}")
    log_dir = osp.join(log_dir, f"interactions_{domain_to_epoch(variant['domain']) + 10}k")
    log_dir = osp.join(log_dir, variant['exp_mode'])
    log_dir = osp.join(log_dir, f"goal_{variant['goal_id']}")

    if should_include_base_log_dir:
        log_dir = osp.join(args.base_log_dir, log_dir)

    # log_dir = osp.join(log_dir, create_simple_exp_name())

    print(log_dir)
    return log_dir


if __name__ == "__main__":

    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau_qf=5e-3,
            soft_target_tau_policy=1e-1,
            target_update_period=1,
            lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        optimistic_exp={},
        candidate_size=10,
        policy_hidden_sizes=[256, 256],
        Qs_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    )

    assert len(variant['Qs_hidden_sizes']) == 9

    assert variant['trainer_kwargs']['soft_target_tau_qf'] == 5e-3
    assert variant['trainer_kwargs']['soft_target_tau_policy'] == 1e-1
    assert variant['trainer_kwargs']['lr'] == 3e-4

    # First generate goal_vel set of size 64 using fixed random seed 1337

    args = get_cmd_args()

    param_module = importlib.import_module(f'configs.{args.config}')
    params = param_module.params

    domain = params['domain']
    exp_mode = params['exp_mode']
    max_path_length = params['max_path_length']
    task = params['task']

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    _, _, wd_goals, _ = pickle.load(open(filename, 'rb'))

    goals = wd_goals

    variant['domain'] = domain
    variant['exp_mode'] = exp_mode
    variant['task'] = task

    variant['goal'] = goals[args.goal]
    variant['goal_id'] = args.goal
    variant['algorithm_kwargs']['max_path_length'] = max_path_length

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(domain)
    print('num epoch: ' + str(domain_to_epoch(domain)))

    variant['seed'] = args.seed

    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0
    variant['optimistic_exp']['beta_UB'] = args.beta_UB
    variant['optimistic_exp']['delta'] = args.delta

    variant['log_dir'] = get_log_dir(variant)
     
    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None
    
    run_experiment_here(experiment, variant,
                        seed=args.seed,
                        use_gpu=not args.no_gpu and torch.cuda.is_available(),
                        gpu_id=gpu_id,

                        # Save the params every snapshot_gap and override previously saved result
                        snapshot_gap=100,
                        snapshot_mode='gap_and_final',

                        log_dir=variant['log_dir']
                    )

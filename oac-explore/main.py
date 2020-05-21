import os
import os.path as osp

import pickle
import argparse
import torch
import importlib
import numpy as np

import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic
from trainer.trainer import SACTrainer
from networks import FlattenMlp
from rl_algorithm import BatchRLAlgorithm

import ray
import logging
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.WARNING,
    webui_host='127.0.0.1',

    # The amount of memory (in bytes)
    object_store_memory=1073741824 * 4, # 1g
    redis_max_memory=1073741824 * 4 # 1g
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

    # Get producer function for policy and value functions
    M = variant['layer_size']

    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024])
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=[M, M])
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
        q_producer,
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

    if prev_exp_state is not None:

        expl_path_collector.restore_from_snapshot(
            prev_exp_state['exploration'])

        ray.get([remote_eval_path_collector.restore_from_snapshot.remote(
            prev_exp_state['evaluation_remote'])])
        ray.get([remote_eval_path_collector.set_global_pkg_rng_state.remote(
            prev_exp_state['evaluation_remote_rng_state']
        )])

        replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])

        trainer.restore_from_snapshot(prev_exp_state['trainer'])

        set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

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
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            min_num_steps_before_training=10000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        optimistic_exp={},
    )

    # First generate goal_vel set of size 64 using fixed random seed 1337

    args = get_cmd_args()

    param_module = importlib.import_module(f'configs.{args.config}')
    params = param_module.params

    domain = params['domain']
    exp_mode = params['exp_mode']
    max_path_length = params['max_path_length']
    task = params['task']

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    idx_list, train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))

    if task =='train':
        goals = train_goals
    elif task =='wd':
        goals = wd_goals
    elif task == 'ood':
        goals = ood_goals
    else:
        raise NotImplementedError
    
    variant['domain'] = domain
    variant['exp_mode'] = exp_mode
    variant['task'] = task

    variant['goal'] = goals[args.goal]
    variant['goal_id'] = idx_list[args.goal]
    variant['algorithm_kwargs']['max_path_length'] = max_path_length
    

    print('domain: ' + domain)
    print('Task goal: ' + str(goals[args.goal]))

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(domain)
    print('num epoch: ' + str(variant['algorithm_kwargs']['num_epochs']))

    variant['seed'] = args.seed
    variant['algorithm_kwargs']['max_path_length'] = max_path_length
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop

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

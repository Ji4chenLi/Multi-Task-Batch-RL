import os
import os.path as osp
import pickle
import argparse
import json
import torch
import torch.nn as nn
import numpy as np

import utils.pytorch_util as ptu
from utils.logging import logger, setup_logger, safe_json, dict_to_safe_json, log_git_infos
from utils.env_utils import env_producer, domain_to_epoch
from utils.rng import set_seed
from utils.pytorch_util import set_gpu_mode
from utils.pythonplusplus import load_gzip_pickle, load_pkl, dump_pkl

from ensemble import EnsemblePredictor
from prob_context_encoder import ProbabilisticContextEncoder
from path_collector import RemotePathCollector
from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from trainer import SuperQTrainer
from networks import FlattenMlp, MlpEncoder, PerturbationGenerator, VaeDecoder
from rl_algorithm import BatchMetaRLAlgorithm
import importlib

import datetime
import dateutil.tz

import socket

import logging 
import ray
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.INFO,
    webui_host='127.0.0.1',

    # Need to set redis memory for
    # fast loading of buffer.
    # Not sure why.
    redis_max_memory=34359738368, #34 Gb
)

def get_log_dir(variant):
    """ Adapted from https://github.com/katerakelly/oyster """
    def create_simple_exp_name():
        """
        Create a unique experiment name with a timestamp
        """
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        return timestamp

    log_dir = 'test_output'
    log_dir = osp.join(log_dir, variant['domain'])
    log_dir = osp.join(log_dir, f"mode_{variant['exp_mode']}")
    log_dir = osp.join(log_dir, f"max_path_length_{variant['max_path_length']}")
    log_dir = osp.join(log_dir, f"count_{variant['algo_params']['num_tasks']}")
    log_dir = osp.join(log_dir, f"seed_{variant['seed']}")
    # log_dir = osp.join(log_dir, create_simple_exp_name())

    log_dir = osp.join(variant['base_log_dir'], log_dir)

    return log_dir

def experiment(variant, bcq_policies, bcq_buffers, prev_exp_state=None):
    # Create the multitask replay buffer based on the buffer list
    train_buffer = MultiTaskReplayBuffer(
        bcq_buffers_list=bcq_buffers,
    )
    # create multi-task environment and sample tasks
    env = env_producer(variant['domain'], variant['seed'])

    env_max_action = float(env.action_space.high[0])
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    vae_latent_dim = 2 * action_dim
    mlp_enconder_input_size = 2 * obs_dim + action_dim + 1 if variant['use_next_obs_in_context'] else obs_dim + action_dim + 1

    variant['env_max_action'] = env_max_action
    variant['obs_dim'] = obs_dim
    variant['action_dim'] = action_dim

    variant['mlp_enconder_input_size'] = mlp_enconder_input_size

    # instantiate networks

    mlp_enconder = MlpEncoder(
        hidden_sizes=[200, 200, 200],
        input_size=mlp_enconder_input_size,
        output_size=2 * variant['latent_dim']
    )
    context_encoder = ProbabilisticContextEncoder(
        mlp_enconder,
        variant['latent_dim']
    )

    Qs = FlattenMlp(
        hidden_sizes=variant['Qs_hidden_sizes'],
        input_size=obs_dim + action_dim + variant['latent_dim'],
        output_size=1,
    )
    vae_decoder = VaeDecoder(
        max_action=env_max_action,
        hidden_sizes=variant['vae_hidden_sizes'],
        input_size=obs_dim + vae_latent_dim + variant['latent_dim'],
        output_size=action_dim,
    )
    perturbation_generator = PerturbationGenerator(
        max_action=env_max_action,
        hidden_sizes=variant['perturbation_hidden_sizes'],
        input_size=obs_dim + action_dim + variant['latent_dim'],
        output_size=action_dim,
    )
    trainer = SuperQTrainer(
        bcq_policies=bcq_policies,
        nets=[context_encoder, Qs, vae_decoder, perturbation_generator],
        num_trans_context=variant['num_trans_context']
    )
    
    path_collector = RemotePathCollector(variant)

    algorithm = BatchMetaRLAlgorithm(
        trainer,
        path_collector,
        train_buffer,
        **variant['algo_params'],
    )

    algorithm.to(ptu.device)

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    # Log the variant
    logger.log("Variant:")
    logger.log(json.dumps(dict_to_safe_json(variant), indent=2))

    algorithm.train(start_epoch)


if __name__ == "__main__":

    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overridden
    # the corresponding attribute in variant

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ant-dir')
    parser.add_argument('--data_models_root', default='../data_and_trained_models')
    
    args = parser.parse_args()

    if args.config:
        variant_module = importlib.import_module(f'configs.{args.config}')
        variant = variant_module.variant
    else:
        raise NotImplementedError

    domain = variant['domain']
    seed = variant['seed']
    exp_mode = variant['exp_mode']
    max_path_length = variant['max_path_length']
    bcq_interactions = variant['bcq_interactions']
    num_tasks = variant['algo_params']['num_tasks']

    if torch.cuda.is_available():
        variant['algo_params']['num_workers'] = int(torch.cuda.device_count() // 0.4)

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    idx_list, train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))

    idx_list = idx_list[:num_tasks]

    variant['algo_params']['train_goals'] = train_goals[:num_tasks]
    variant['algo_params']['wd_goals'] = wd_goals
    variant['algo_params']['ood_goals'] = ood_goals

    # Directory to the buffers, trained policies and ensemble_params

    sub_dir =  f"buffers/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}"
    buffer_dir = osp.join(args.data_models_root, sub_dir)
    
    buffer_dir = osp.join(args.data_models_root, f"buffers/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}")
    policy_dir = osp.join(args.data_models_root, f"policies/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}")
    ensemble_params_dir = osp.join(args.data_models_root, f"reward_prediction_ensemble/{domain}/mode_{exp_mode}/max_path_length_{max_path_length}/goal_")
    
    # Load policy
    bcq_policies = []
    for idx in idx_list:
        pname = f'policy_goal_0{idx}' if idx < 10  else f'policy_goal_{idx}'
        filename = os.path.join(policy_dir, pname)
        with open(filename, 'rb') as f:
            bcq_policies.append(pickle.load(f))
            f.close()
    
    assert len(bcq_policies) == len(idx_list)

    # Load buffer
    bcq_buffers = []

    buffer_loader_id_list = []
    for i, idx in enumerate(idx_list):
        bname = f'goal_0{idx}.zip_pkl' if idx < 10 else f'goal_{idx}.zip_pkl'
        filename = os.path.join(buffer_dir, bname)
        rp_buffer = ReplayBuffer.remote(
            index=i,
            seed=seed,
            num_trans_context=variant['num_trans_context'],
            in_mdp_batch_size=variant['in_mdp_batch_size'],
            use_next_obs_in_context=variant['use_next_obs_in_context'],
        )
        
        buffer_loader_id_list.append(rp_buffer.load_from_gzip.remote(filename))
        bcq_buffers.append(rp_buffer)
    ray.get(buffer_loader_id_list)

    assert len(bcq_buffers) == len(idx_list)

    # set up logger 
    variant['log_dir'] = get_log_dir(variant)

    logger.reset()
    setup_logger(log_dir=variant['log_dir'], snapshot_gap=100, snapshot_mode="gap")

    logger.log(f"Seed: {seed}")
    set_seed(seed)

    logger.log(f'Using GPU: {True}')
    set_gpu_mode(mode=True, gpu_id=0)

    experiment(variant, bcq_policies, bcq_buffers, prev_exp_state=None)


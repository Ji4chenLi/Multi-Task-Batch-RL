import os
import os.path as osp
import pickle
import argparse
import json
import torch
import numpy as np
import importlib
import gtimer as gt

import utils.pytorch_util as ptu
from utils.logging import logger, setup_logger, safe_json, dict_to_safe_json, log_git_infos
from utils.env_utils import env_producer, domain_to_epoch
from utils.rng import set_seed
from utils.pytorch_util import set_gpu_mode
from utils.pythonplusplus import load_gzip_pickle, load_pkl, dump_pkl

from path_collector import RemotePathCollector
from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from trainer import SuperQTrainer
from networks import FlattenMlp, MlpEncoder, PerturbationGenerator, VaeDecoder
from rl_algorithm import BatchMetaRLAlgorithm

import socket

import datetime
import dateutil.tz

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
    redis_max_memory= 1073741824 * 2, # 2 Gb
)

def experiment(variant, bcq_buffers, prev_exp_state=None):
    # Create the multitask replay buffer based on the buffer list
    train_buffer = MultiTaskReplayBuffer(
        bcq_buffers_list=bcq_buffers,
    )
    # create multi-task environment and sample tasks
    env = env_producer(variant['domain'], variant['seed'])
    env.reset()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    network_ensemble = []
    for _ in range(variant['num_network_ensemble']):
        P = FlattenMlp(
            hidden_sizes=variant['P_hidden_sizes'],
            input_size=obs_dim + action_dim,
            output_size=obs_dim,
        )
        network_ensemble.append(P)

    trainer = SuperQTrainer(
        env,
        network_ensemble=network_ensemble,
        train_goal=variant['train_goal'],
        std_threshold=variant['std_threshold']
    )

    algorithm = BatchMetaRLAlgorithm(
        trainer,
        train_buffer,
        **variant['algo_params'],
    )

    algorithm.to(ptu.device)

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)


def get_log_dir(variant):

    # def create_simple_exp_name():
    #     """
    #     Create a unique experiment name with a timestamp
    #     """
    #     now = datetime.datetime.now(dateutil.tz.tzlocal())
    #     timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    #     return timestamp

    # log_dir = 'test_output_seperate_optimizer'
    log_dir = './test'
    log_dir = osp.join(log_dir, variant['domain'])
    log_dir = osp.join(log_dir, f"mode_{variant['exp_mode']}")
    log_dir = osp.join(log_dir, f"max_path_length_{variant['max_path_length']}")
    log_dir = osp.join(log_dir, f"goal_{variant['algo_params']['train_goal_id']}")

    log_dir = osp.join(variant['base_log_dir'], log_dir)

    return log_dir



if __name__ == "__main__":

    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overridden
    # the corresponding attribute in variant

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='walker-param')
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

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    idx_list, train_goals, _, _ = pickle.load(open(filename, 'rb'))

    idx_list = idx_list[:num_tasks]

    variant['idx_list'] = idx_list

    # Directory to the buffers and trained policies
    sub_buffer_dir = f"buffers_qpos_qvel/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}"
    buffer_dir = osp.join(args.data_models_root, sub_buffer_dir)

    # Load buffer
    bcq_buffers = []

    buffer_loader_id_list = []
    for i, bname in enumerate(sorted(os.listdir(buffer_dir))):
        if i in idx_list:
            rp_buffer = ReplayBuffer.remote(
                index=i,
                seed=variant['seed'],
                in_mdp_batch_size=variant['in_mdp_batch_size'],
            )
            filename = os.path.join(buffer_dir, bname)
            buffer_loader_id_list.append(rp_buffer.load_from_gzip.remote(filename))
            bcq_buffers.append(rp_buffer)
    ray.get(buffer_loader_id_list)

    start = variant['start']
    end = variant['end']
    for i in range(start, end):
        variant['algo_params']['train_goal_id'] = i
        variant['train_goal'] = train_goals[i]

        # set up logger
        variant['log_dir'] = get_log_dir(variant)

        logger.reset()
        setup_logger(log_dir=variant['log_dir'], snapshot_gap=100, snapshot_mode="gap")

        # Log the variant
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))

        logger.log(f'Seed: {seed}')
        set_seed(seed)

        logger.log(f'Using GPU: {True}')
        set_gpu_mode(mode=True, gpu_id=0)

        gt.reset()

        experiment(variant, bcq_buffers, prev_exp_state=None)


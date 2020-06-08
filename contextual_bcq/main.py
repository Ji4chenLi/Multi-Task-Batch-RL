import os
import os.path as osp
import pickle
import json
import numpy as np
import torch
import argparse
import os
import datetime
import dateutil.tz
import importlib
from BCQ_plus_encoder import BCQ

from utils.env_utils import env_producer, domain_to_num_goals
import utils.pytorch_util as ptu
from utils.pytorch_util import set_gpu_mode
import gtimer as gt

from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from path_collector import RemotePathCollector
from rl_alogrithm import BatchMetaRLAlgorithm

from utils.logging import logger, setup_logger, safe_json, dict_to_safe_json, log_git_infos
from utils.rng import set_seed

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

    def create_simple_exp_name():
        """
        Create a unique experiment name with a timestamp
        """
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        return timestamp

    # log_dir = 'test_output_seperate_optimizer'
    log_dir = 'test_output'
    log_dir = osp.join(log_dir, variant['domain'])
    log_dir = osp.join(log_dir, f"mode_{variant['exp_mode']}")
    log_dir = osp.join(log_dir, f"max_path_length_{variant['max_path_length']}")
    log_dir = osp.join(log_dir, f"count_{variant['algo_params']['num_tasks']}")
    log_dir = osp.join(log_dir, f"seed_{variant['seed']}")
    # log_dir = osp.join(log_dir, create_simple_exp_name())

    log_dir = osp.join(variant['base_log_dir'], log_dir)

    return log_dir


if __name__ == "__main__":
	
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

	variant['idx_list'] = idx_list

	# Directory to the buffers, trained policies and ensemble_params
	sub_buffer_dir = f"../data_and_trained_models/buffers/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}"
	buffer_dir = osp.join(args.data_models_root, sub_buffer_dir)

	print("Buffer directory: " + buffer_dir)

	variant['algo_params']['train_goals'] = train_goals[:num_tasks]
	variant['algo_params']['wd_goals'] = wd_goals
	variant['algo_params']['ood_goals'] = ood_goals

	# set up logger 
	variant['log_dir'] = get_log_dir(variant)
	logger.reset()
	setup_logger(log_dir = variant['log_dir'], snapshot_gap=100, snapshot_mode="gap")

	logger.log(f'Seed: {seed}')
	set_seed(seed)

	logger.log(f'Using GPU: {True}')
	set_gpu_mode(mode=True, gpu_id=0)

	# Get the information of the environment
	env = env_producer(domain, seed)

	state_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	max_action = float(env.action_space.high[0])

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
		)
		
		buffer_loader_id_list.append(rp_buffer.load_from_gzip.remote(filename))
		bcq_buffers.append(rp_buffer)
	ray.get(buffer_loader_id_list)

	assert len(bcq_buffers) == len(idx_list)

	train_buffer = MultiTaskReplayBuffer(
		bcq_buffers_list=bcq_buffers,
	)
	path_collector = RemotePathCollector(variant)

	# Initialize policy
	policy = BCQ(state_dim, action_dim, max_action, **variant['policy_params'])

	algorithm = BatchMetaRLAlgorithm(
		policy,
		path_collector,
		train_buffer,
		**variant['algo_params']
	)

	algorithm.to(ptu.device)

	# Log the variant
	logger.log("Variant:")
	logger.log(json.dumps(dict_to_safe_json(variant), indent=2))

	algorithm.train()
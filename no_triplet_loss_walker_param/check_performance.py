import os
import os.path as osp
import pickle
import argparse
import json
import torch
import numpy as np
import importlib
from collections import OrderedDict

import utils.pytorch_util as ptu
from utils.logging import logger, setup_logger, safe_json, dict_to_safe_json, log_git_infos
from utils.env_utils import env_producer, domain_to_epoch, domain_to_num_goals
from utils.rng import set_seed
from utils.pytorch_util import set_gpu_mode
from utils.pythonplusplus import load_gzip_pickle, load_pkl, dump_pkl

from path_collector import RemotePathCollector
from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from trainer import SuperQTrainer
from networks import FlattenMlp, MlpEncoder, PerturbationGenerator, VaeDecoder
from rl_algorithm import BatchMetaRLAlgorithm

import datetime
import dateutil.tz
import gzip

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ant-dir-normal-200-32tasks')
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
    num_tasks = variant['algo_params']['num_tasks']

    assert args.config == f'{domain}-{exp_mode}-{max_path_length}-{num_tasks}tasks'

    if torch.cuda.is_available():
        variant['algo_params']['num_workers'] = int(torch.cuda.device_count() // 0.4) - 1
        num_workers = variant['algo_params']['num_workers']

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))

    env = env_producer(variant['domain'], variant['seed'])
    ob = env.reset()
    env_max_action = float(env.action_space.high[0])
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    set_seed(seed)
    set_gpu_mode(mode=True, gpu_id=0)

    variant['env_max_action'] = env_max_action
    variant['obs_dim'] = obs_dim
    variant['action_dim'] = action_dim

    variant['num_evals'] = 4

    path_collector = RemotePathCollector(variant)

    logger.reset()
    setup_logger(log_dir='./check_performane_loggings/check_direction')

    for itr in [25, 50, 75, 100, 125, 150, 175, 200]:
        path_to_snapshot = f'/hdd/jiachen/tiMe_prob_encoder_results/{domain}/max_path_length_{max_path_length}/count_{num_tasks}/itr_{itr}.zip_pkl'
        ss = load_gzip_pickle(path_to_snapshot)
        ss = ss['trainer']

        ss_encoder_state_dict = OrderedDict()
        for key, value in ss['context_encoder_state_dict'].items():
            if 'mlp_encoder' in key:
                ss_encoder_state_dict[key.replace('mlp_encoder.', '')] = value

        params_list = [
            ptu.gpu_dict_to_cpu(ss_encoder_state_dict), ptu.gpu_dict_to_cpu(ss['Qs_state_dict']),
            ptu.gpu_dict_to_cpu(ss['vae_decoder_state_dict']), ptu.gpu_dict_to_cpu(ss['perturbation_generator_dict'])
        ]
        path_collector.set_network_params(params_list)


        evaluation_train_obj_id_list = []
        count = 0
        while count < len(train_goals) :
            if len(train_goals) - count < num_workers:
                evaluation_obj_id = path_collector.async_evaluate_test(train_goals[count:])
                count = len(train_goals)
            else:
                evaluation_obj_id = path_collector.async_evaluate_test(train_goals[count:count + num_workers])
                count += num_workers
            evaluation_train_obj_id_list.append(evaluation_obj_id)

        evaluation_wd_obj_id_list = []
        count = 0
        while count < len(wd_goals) :
            if len(wd_goals) - count < num_workers:
                evaluation_obj_id = path_collector.async_evaluate_test(wd_goals[count:])
                count = len(wd_goals)
            else:
                evaluation_obj_id = path_collector.async_evaluate_test(wd_goals[count:count + num_workers])
                count += num_workers
            evaluation_wd_obj_id_list.append(evaluation_obj_id)
        
        evaluation_ood_obj_id_list = []
        count = 0
        while count < len(ood_goals) :
            if len(ood_goals) - count < num_workers:
                evaluation_obj_id = path_collector.async_evaluate_test(ood_goals[count:])
                count = len(ood_goals)
            else:
                evaluation_obj_id = path_collector.async_evaluate_test(ood_goals[count:count + num_workers])
                count += num_workers
            evaluation_ood_obj_id_list.append(evaluation_obj_id)

        eval_train_returns = []
        for obj_id in evaluation_train_obj_id_list:
            eval_train_returns.extend(ray.get(obj_id))

        eval_wd_returns = []
        for obj_id in evaluation_wd_obj_id_list:
            eval_wd_returns.extend(ray.get(obj_id))

        eval_ood_returns = []
        for obj_id in evaluation_ood_obj_id_list:
            eval_ood_returns.extend(ray.get(obj_id))

        # avg_train_episode_returns = [item[0] for item in eval_train_returns]
        # final_train_achieved = [item[1] for item in eval_train_returns]
        # train_avg_returns =  [np.mean([r[i] for r in avg_train_episode_returns]) for i in range(variant['num_evals'])]

        # avg_wd_episode_returns = [item[0] for item in eval_wd_returns]
        # final_wd_achieved = [item[1] for item in eval_wd_returns]
        # wd_avg_returns = [np.mean([r[i] for r in avg_wd_episode_returns]) for i in range(variant['num_evals'])]

        # avg_ood_episode_returns = [item[0] for item in eval_ood_returns]
        # final_ood_achieved = [item[1] for item in eval_ood_returns]
        # ood_avg_returns = [np.mean([r[i] for r in avg_ood_episode_returns]) for i in range(variant['num_evals'])]

        # eval = OrderedDict()

        # eval['avg_train_episode_returns'] = avg_train_episode_returns
        # eval['final_train_achieved'] = final_train_achieved
        # eval['train_avg_returns'] = train_avg_returns

        # eval['avg_wd_episode_returns'] = avg_wd_episode_returns
        # eval['final_wd_achieved'] = final_wd_achieved
        # eval['wd_avg_returns'] = wd_avg_returns

        # eval['avg_ood_episode_returns'] = avg_ood_episode_returns
        # eval['final_ood_achieved'] = final_ood_achieved
        # eval['ood_avg_returns'] = ood_avg_returns

        online_train_achieved = [item for item in eval_train_returns]

        online_wd_achieved = [item for item in eval_wd_returns]

        online_ood_achieved = [item for item in eval_ood_returns]

        eval = OrderedDict()

        # eval['avg_train_episode_returns'] = avg_train_episode_returns
        eval['online_train_achieved'] = online_train_achieved

        # eval['avg_wd_episode_returns'] = avg_wd_episode_returns
        eval['online_wd_achieved'] = online_wd_achieved

        # eval['avg_ood_episode_returns'] = avg_ood_episode_returns
        eval['online_ood_achieved'] = online_ood_achieved


        logger.log("Epoch {} finished".format(itr), with_timestamp=True)
        logger.record_dict(eval, prefix='eval/')

        write_header = True if itr == 25 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)
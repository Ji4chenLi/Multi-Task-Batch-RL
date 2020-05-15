import os
import os.path as osp
import pickle
import json
import numpy as np
import torch
import argparse
import os
import ray
import logging
import datetime
import dateutil.tz
from BCQ_plus_encoder import BCQ

from env_utils import env_producer, domain_to_num_goals
import utils.pytorch_util as ptu
from utils.pytorch_util import set_gpu_mode
import gtimer as gt

from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from path_collector import RemotePathCollector
from rl_alogrithm import BatchMetaRLAlgorithm
from utils.pythonplusplus import load_gzip_pickle

from utils.logging import logger, setup_logger, safe_json, dict_to_safe_json, log_git_infos
from utils.rng import set_seed

ray.init(
	# If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.WARNING,

    # The amount of memory (in bytes)
    object_store_memory=1073741824 * 8, # 6g
    redis_max_memory=1073741824 * 8 # 6g
)	


if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="halfcheetah-vel")
    parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--goal', default=0, type=int)
    parser.add_argument('--num_tasks', default=32, type=int)
    parser.add_argument('--meta_batch_size', default=32, type=int)
    parser.add_argument("--buffer_type", default="Robust")				# Prepends name to filename.
    parser.add_argument("--num_train_loops_per_epoch", default=100, type=int)			# How often (time steps) we evaluate
    parser.add_argument("--num_epoches", default=3e4, type=float)		# Max time steps to run environment for
    parser.add_argument("--exp_mode", default="hard", type=str)
    parser.add_argument("--hyper_params_dir", default="hyperparams_bcq_encoder_match_tiMe.pkl", type=str)
    parser.add_argument("--param_id", default=0, type=int)

    args = parser.parse_args()

    # load hyper parameters 
with open(args.hyper_params_dir, 'rb') as f:
    hyper_params = pickle.load(f)[args.param_id]

    network_params, vae_latent_dim_multiplicity, target_q_coef = hyper_params
    actor_hid_sizes, critic_hid_sizes, vae_e_hid_sizes, vae_d_hid_sizes = network_params

    variant = dict(
        policy_params=dict(
            vae_latent_dim_multiplicity=vae_latent_dim_multiplicity, 
            target_q_coef=target_q_coef,
            actor_hid_sizes=actor_hid_sizes, 
            critic_hid_sizes=critic_hid_sizes, 
            vae_e_hid_sizes=vae_e_hid_sizes, 
            vae_d_hid_sizes=vae_d_hid_sizes,
            encoder_latent_dim=20, 
            g_hid_sizes=[512, 512, 512, 512],
            g_latent_dim=10,
            h_hid_sizes=[512, 512, 512],
            E_hid_sizes=[512, 512, 512, 512, 512, 512, 512],
            P_hid_sizes=[64],
        ),
        algo_params=dict(
            num_epochs=200,
            num_train_loops_per_epoch=args.num_train_loops_per_epoch,
            num_tasks=args.num_tasks,
            meta_batch_size=args.meta_batch_size,
            use_same_context=False,
        ),
        domain=args.domain,
        seed=args.seed,
        num_trans_context=20,
        in_mdp_batch_size=128,
        num_evals=5,
        max_path_length=1000,
    )
    print(f'actor hid sizes: {actor_hid_sizes}')
    print(f'critic hid sizes: {critic_hid_sizes}')
    print(f'vae e hid sizes: {vae_e_hid_sizes}')
    print(f'vae d hid sizes: {vae_d_hid_sizes}')

    print("---------------------------------------")
    print("Eval frequencies: " + str(args.num_train_loops_per_epoch))
    print("Num epochs: " + str(variant['algo_params']['num_epochs']))
    print("Hyper param dir: " + args.hyper_params_dir)
    print("VAE latent dim multiplicity and target q coef: " + str(vae_latent_dim_multiplicity) + " and " + str(target_q_coef))
    print("---------------------------------------")

    env = env_producer(args.domain, args.seed, 1.0)

    state_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    max_action = float(env.action_space.high[0])
    

    # Initialize policy
    policy = BCQ(state_dim, action_dim, max_action, **variant['policy_params'])
    # ss = load_gzip_pickle(f'./result_params/params_{args.exp_mode}.zip_pkl')
    ss_path = '/home/jiachen/bcq_encoder/test_output/halfcheetah-vel/seed_0/num_tasks_32/mode_hard/max_path_length_1000/2020_03_16_16_46_37/params.zip_pkl'
    ss = load_gzip_pickle(ss_path)
    policy.restore_from_snapshot(ss)

    path_collector = RemotePathCollector.remote(variant, eval_goals_set=[1.0] * 1)

    for t in range(3):
        if t == 0:
            np.random.seed(1337)
            if args.exp_mode == 'hard':
                goals = np.random.uniform(0, 1.5, size=(domain_to_num_goals(args.domain),))
            elif args.exp_mode == 'medium':
                goals = np.random.uniform(0, 2.5, size=(domain_to_num_goals(args.domain),))
            else:
                goals_left = np.random.uniform(0.0, 1.5, size=(domain_to_num_goals(args.domain) // 2,))
                goals_right = np.random.uniform(2.5, 3.0, size=(domain_to_num_goals(args.domain) // 2,))
                goals = np.concatenate((goals_left, goals_right))
            print("---------------------------------------")
            print("Meta training")
            print("---------------------------------------")
        elif t == 1:
            np.random.seed(1337)
            if args.exp_mode == 'interpolate':
                goals = np.random.uniform(1.5, 2.5, size=(domain_to_num_goals(args.domain),))
            else:
                goals = np.random.uniform(2.5, 3.0, size=(domain_to_num_goals(args.domain),))
            print("---------------------------------------")
            print("OOD meta testing training")
            print("---------------------------------------")
        else:
            np.random.seed(2337)
            if args.exp_mode == 'hard':
                goals = np.random.uniform(0, 1.5, size=(domain_to_num_goals(args.domain),))
            elif args.exp_mode == 'medium':
                goals = np.random.uniform(0, 2.5, size=(domain_to_num_goals(args.domain),))
            else:
                goals_left = np.random.uniform(0.0, 1.5, size=(domain_to_num_goals(args.domain) // 2,))
                goals_right = np.random.uniform(2.5, 3.0, size=(domain_to_num_goals(args.domain) // 2,))
                goals = np.concatenate((goals_left, goals_right))
            print("---------------------------------------")
            print("WD meta testing training")
            print("---------------------------------------")

        eval_goals_set = [vel for vel in goals]

        print(eval_goals_set)

        set_seed(args.seed)

        set_gpu_mode(mode=True, gpu_id=0)

        # Get the information of the environment
        env = env_producer(args.domain, args.seed, 1.0)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.low.size
        action_dim = env.action_space.low.size
        max_action = float(env.action_space.high[0])

        params_list = []
        for net in policy.eval_networks:
            params_list.append(ptu.state_dict_cpu(net))

        ray.get(path_collector.set_policy_params.remote(params_list))

        evaluation_train_obj_id_list = [path_collector.async_evaluate.remote([eval_goals_set[0]])]

        # evaluation_train_obj_id_list = []
        # for i in range(len(eval_goals_set) // 8):
        #     evaluation_obj_id = path_collector.async_evaluate.remote(eval_goals_set[i * 4 : (i+1) * 4])
        #     evaluation_train_obj_id_list.append(evaluation_obj_id)
        
        eval_train_returns = []
        for obj_id in evaluation_train_obj_id_list:
            eval_train_returns.extend(ray.get(obj_id))
        
    

        avg_episode_returns = [item[0] for item in eval_train_returns]
        final_achieved = [item[1] for item in eval_train_returns]
        # final_velocities = [item[2][0] for item in eval_train_returns]
        avg_returns = np.mean(avg_episode_returns)

        print(avg_episode_returns)
        print(final_achieved)
        print(avg_returns)



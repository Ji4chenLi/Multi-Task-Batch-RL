import gym
import numpy as np
import torch
import random

import gzip
import argparse
import os
import importlib

import utils
import BCQ
from env_utils import env_producer, domain_to_num_goals, domain_to_epoch
import pickle

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10, max_path_length=200):

	episode_returns = []

	for _ in range(eval_episodes):

		obs = env.reset()
		done = False
		path_length = 0

		episode_ret = 0

		while not done and path_length < max_path_length:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)

			episode_ret += reward
			
			path_length += 1

		episode_returns.append(episode_ret)

	mean = np.mean(episode_returns)
	std = np.std(episode_returns)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes. mean {mean}, std {std}")
	print("---------------------------------------")
	return mean, std


def train_loop():
	training_iters = 0
	save_root_dir = f"./results/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{interactions}k/seed_{args.seed}/goal_{goal_id}/"
	if not os.path.exists(save_root_dir):
		os.makedirs(save_root_dir)

	num = 0
	best_mean = -1e6
	best_index = 0
	mean_list = []
	std_list = []
	while training_iters < max_timesteps: 
		policy.train(replay_buffer, iterations=int(args.eval_freq))

		mean, std = evaluate_policy(policy, eval_episodes=args.eval_episodes, max_path_length=max_path_length)

		if mean > best_mean:
			best_index = num
			best_mean = mean

		mean_list.append(mean)
		std_list.append(std)

		np.save(save_root_dir + "mean.npy", np.array(mean_list))
		np.save(save_root_dir + "std.npy", np.array(std_list))

		if not os.path.exists(save_root_dir + "policies"): 
			os.makedirs(save_root_dir + "policies")
		policy_file_name = f"policies/policy_num_{num}"
		pickle.dump(policy, open(save_root_dir + policy_file_name, "wb"))

		num += 1
		training_iters += args.eval_freq

		print("Training iterations: " + str(training_iters), flush=True)

	print(f'Save the best policy, with the best index: {best_index}')
	
	policies_root = f"./policies/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{interactions}k/seed_{args.seed}"
	if not os.path.exists(policies_root): 
		os.makedirs(policies_root)

	best_policy_fname = save_root_dir + f"policies/policy_num_{best_index}"
	with open(best_policy_fname, 'rb') as f:
            best_policy = pickle.load(f)

	if goal_id < 10:
		policy_file_name = policies_root + f"/policy_goal_0{goal_id}"
	else:
		policy_file_name = policies_root + f"/policy_goal_{goal_id}"
	pickle.dump(best_policy, open(policy_file_name, "wb"))



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument('--goal', default=0, type=int)
	parser.add_argument('--eval_episodes', default=10, type=int)
	parser.add_argument('--config', type=str, default='ant-dir')
	parser.add_argument("--eval_freq", default=5000, type=float)			# How often (time steps) we evaluate
	parser.add_argument('--oac_folder_name', type=str, default='../oac-explore')
	
	args = parser.parse_args()

	param_module = importlib.import_module(f'configs.{args.config}')
	params = param_module.params

	domain = params['domain']
	exp_mode = params['exp_mode']
	max_path_length = params['max_path_length']
	interactions = params['interactions']
	task = params['task']
	max_timesteps = params['max_timesteps']
	hypter_param_dir = params['hypter_param_dir']

	filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
	idx_list, train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))

	goal_id = idx_list[args.goal]

	if interactions == -1:
		interactions = domain_to_epoch(domain) + 10
		buffer_file_name = '/params.zip_pkl'
	else:
		buffer_file_name = f'/itr_{interactions}.zip_pkl'

	assert args.config == f'{domain}-{exp_mode}-{max_path_length}'

	folder_name = f'oac-explore/data/{task}_task_results/{domain}/seed_{args.seed}/max_path_length_{max_path_length}/interactions_{domain_to_epoch(domain) + 10}k/{exp_mode}/goal_{goal_id}'
	folder_name = os.path.join(args.oac_folder_name, folder_name)

	buffer_name = folder_name + buffer_file_name

	print('---------------------------------')
	print(buffer_name)
	print('---------------------------------')

	if task =='train':
		goals = train_goals
	elif task =='wd':
		goals = wd_goals
	elif task == 'ood':
		goals = ood_goals
	else:
		raise NotImplementedError
	
	# load hyper parameters 
	with open(hypter_param_dir, 'rb') as f:
		hyper_params = pickle.load(f)[0]

	network_params, latent_dim_multiplicity, target_q_coef = hyper_params
	actor_hid_sizes, critic_hid_sizes, vae_e_hid_sizes, vae_d_hid_sizes = network_params

	print(f'actor hid sizes: {actor_hid_sizes}')
	print(f'critic hid sizes: {critic_hid_sizes}')
	print(f'vae e hid sizes: {vae_e_hid_sizes}')
	print(f'vae d hid sizes: {vae_d_hid_sizes}')

	print("---------------------------------------")
	print("Buffer file: " + buffer_name)
	print("Eval frequencies: " + str(args.eval_freq))
	print("Max time steps: " + str(max_timesteps))
	print("Hypter param dir: " + hypter_param_dir)
	print("Latent dim multiplicity and target q coef: " + str(latent_dim_multiplicity) + " and " + str(target_q_coef))
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	
	env = env_producer(domain, args.seed, goals[args.goal])
	print('env goal vel: ' + str(env._goal))

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	
	state_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	max_action = float(env.action_space.high[0])

	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action, latent_dim_multiplicity, target_q_coef,
					 actor_hid_sizes=actor_hid_sizes, 
					 critic_hid_sizes=critic_hid_sizes, 
					 vae_e_hid_sizes=vae_e_hid_sizes, 
					 vae_d_hid_sizes=vae_d_hid_sizes)

	# Load buffer
	replay_buffer = utils.ReplayBuffer()

	with gzip.open(buffer_name, 'rb') as f:
		buffer = pickle.load(f)['replay_buffer']
		f.close()

	# num_trans = (domain_to_epoch(args.domain) + 10) * 1000
	num_trans = int(interactions * 1000)
	print(f'num_trans: {num_trans}')
	assert num_trans <= int(domain_to_epoch(domain) + 10) * 1000

	obs = buffer['_observations'][:num_trans]
	actions = buffer['_actions'][:num_trans]
	rewards = buffer['_rewards'][:num_trans]
	next_obs = buffer['_next_obs'][:num_trans]
	dones = buffer['_terminals'][:num_trans]
	data = list(zip(obs, next_obs, actions, rewards, dones))

	dest_folder = f'./buffers/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{interactions}k/seed_{args.seed}'
	print(dest_folder)

	if not os.path.exists(dest_folder):
		os.makedirs(dest_folder)
		print('Saving the buffer!')

	if goal_id < 10:
		buffer_name = '/goal_0' + str(goal_id) + '.zip_pkl'
	else:
		buffer_name = '/goal_' + str(goal_id) + '.zip_pkl'

	if not os.path.exists(dest_folder + buffer_name):
		with gzip.open(dest_folder + buffer_name, 'wb') as f:
			pickle.dump(data, f)

		print('Buffer size: ', len(data))
		print('Buffer saved!')
	
	replay_buffer.storage = data

	episode_num = 0
	done = True 

	train_loop()
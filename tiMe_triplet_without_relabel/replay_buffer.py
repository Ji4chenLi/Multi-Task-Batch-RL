import numpy as np
import torch
import os

import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy, get_numpy
from utils.rng import set_seed
from torch.utils.data import Dataset, DataLoader, RandomSampler

import ray
from utils.pythonplusplus import load_gzip_pickle
# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
@ray.remote
class ReplayBuffer(object):
	def __init__(
		self,
		index,
		seed,
		num_trans_context,
		in_mdp_batch_size,
		num_candidate_context=10,
		use_next_obs_in_context=True,
	):
		torch.set_num_threads(1)
		self.storage = []
		self.num_candidate_context = num_candidate_context
		self.num_trans_context = num_trans_context
		self.in_mdp_batch_size = in_mdp_batch_size
		self.use_next_obs_in_context = use_next_obs_in_context

		self.buffer_size = 0
		set_seed(10 * seed + 2337 + index)

	# Expects tuples of (state, next_state, action, reward, done, vel_dif)
	def add(self, data):
		self.storage.append(data)

	def sample(self):
		# Sample context for each task
		context_sample_size = self.num_candidate_context * self.num_trans_context
			
		ind = np.random.randint(
			0, self.buffer_size, size=(context_sample_size, )
		)
		sampled_data = self.storage[ind]
		obs = np.stack(list(sampled_data[:, 0]))
		next_obs = np.stack(list(sampled_data[:, 1]))
		actions = np.stack(list(sampled_data[:, 2]))
		rewards = sampled_data[:, 3].astype(np.float64).reshape(-1, 1)
		
		obs = obs.reshape(self.num_candidate_context, self.num_trans_context, -1)
		actions = actions.reshape(self.num_candidate_context, self.num_trans_context, -1)
		rewards = rewards.reshape(self.num_candidate_context, self.num_trans_context, -1)
		next_obs = next_obs.reshape(self.num_candidate_context, self.num_trans_context, -1)

		if self.use_next_obs_in_context:
			context = np.concatenate([obs, actions, next_obs, rewards], axis=2)
		else:
			assert False
			context = np.concatenate([obs, actions, rewards], axis=2)

		# Sample training transitions
		sample_range = np.delete(np.arange(self.buffer_size), ind)
		ind = np.random.choice(
			sample_range, size=(self.in_mdp_batch_size, )
		)
		sampled_data = self.storage[ind]

		# Construct returning batch using the the sampled data
		batch = []

		# The number "3" here is because the first three part
		# in the buffer state, next_state, action are np.array.
		# But the reward is just number
		for i in range(3):
			batch.append(np.stack(list(sampled_data[:, i])))
		batch.append(sampled_data[:, 3].astype(np.float64).reshape(-1, 1))

		batch.append(context)

		return batch

	def save(self, filename):
		np.save(filename, self.storage)

	def load(self, filename):
		self.storage = np.load(filename, allow_pickle=True)
		self.buffer_size = len(self.storage)

	def load_from_gzip(self, filename):
		self.storage = np.array(load_gzip_pickle(filename))
		self.buffer_size = len(self.storage)

		
class MultiTaskReplayBuffer(object):
	def __init__(
			self,
			bcq_buffers_list,
		):
		torch.set_num_threads(1)
		self.num_tasks = len(bcq_buffers_list)
		self.task_buffers = dict([(idx, buffer) for idx, buffer in enumerate(bcq_buffers_list)])

	def sample_training_data(self, batch_idxes):

		batch_obj_ids = [
			self.task_buffers[idx].sample.remote() for i, idx in enumerate(batch_idxes)
		]
		return batch_obj_ids

	def end_epoch(self, epoch):
		pass


# class ReplayBuffer_local(object):
# 	def __init__(
# 		self,
# 		num_trans_context,
# 		in_mdp_batch_size,
# 	):
# 		self.storage = []
# 		self.num_trans_context = num_trans_context
# 		self.in_mdp_batch_size = in_mdp_batch_size
# 		self.buffer_size = 0

# 	# Expects tuples of (state, next_state, action, reward, done, vel_dif)
# 	def add(self, data):
# 		self.storage.append(data)

# 	def sample(self, use_same_context):
# 		# Sample context for each task
# 		if use_same_context:
# 			context_sample_size = self.num_trans_context
# 		else:
# 			context_sample_size = self.num_trans_context * self.in_mdp_batch_size
			
# 		ind = np.random.randint(
# 			0, self.buffer_size, size=(context_sample_size, )
# 		)
# 		sampled_data = self.storage[ind]
# 		obs = np.stack(list(sampled_data[:, 0]))
# 		actions = np.stack(list(sampled_data[:, 2]))
# 		rewards = sampled_data[:, 3].astype(np.float64).reshape(-1, 1)
		
# 		if use_same_context:
# 			context = np.concatenate([obs[None], actions[None], rewards[None]], axis=2)
# 		else:
# 			obs = obs.reshape(self.in_mdp_batch_size, self.num_trans_context, -1)
# 			actions = actions.reshape(self.in_mdp_batch_size, self.num_trans_context, -1)
# 			rewards = rewards.reshape(self.in_mdp_batch_size, self.num_trans_context, -1)
# 			context = np.concatenate([obs, actions, rewards], axis=2)

# 		# Sample training transitions
# 		sample_range = np.delete(np.arange(self.buffer_size), ind)
# 		ind = np.random.choice(
# 			sample_range, size=(self.in_mdp_batch_size, )
# 		)
# 		sampled_data = self.storage[ind]

# 		# Construct returning batch using the the sampled data
# 		batch = []

# 		# The number "3" here is because the first three part
# 		# in the buffer state, next_state, action are np.array.
# 		# But the reward is just number
# 		for i in range(3):
# 			batch.append(np.stack(list(sampled_data[:, i])))
# 		batch.append(sampled_data[:, 3].astype(np.float64).reshape(-1, 1))

# 		batch.append(context)

# 		return batch

# 	def save(self, filename):
# 		np.save(filename, self.storage)

# 	def load(self, filename):
# 		self.storage = np.load(filename, allow_pickle=True)
# 		self.buffer_size = len(self.storage)

# 	def load_from_gzip(self, filename):
# 		self.storage = np.array(load_gzip_pickle(filename))
# 		self.buffer_size = len(self.storage)



# class TransitionDataset(Dataset):
# 	"""Transition dataset."""

# 	def __init__(self, goal_root_dir, size):
# 		"""
# 		Args:
# 			goal_root_dir (string): Directory with all the transitions.
# 		"""
# 		self.goal_root_dir = goal_root_dir
# 		self.size = size

# 	def __len__(self):
# 		return self.size

# 	def __getitem__(self, idx):
# 		if torch.is_tensor(idx):
# 			idx = idx.tolist()
# 		sub_root = f'{idx // 1000}k/'
# 		trans_name = os.path.join(self.goal_root_dir + sub_root, f'trans_{idx}.npy')
# 		trans = list(np.load(trans_name, allow_pickle=True))
# 		sample = {'trans': trans}

# 		return sample

# @ray.remote(num_cpus=1.0)
# class ReplayBuffer_dataLoader(object):
# 	def __init__(
# 		self,
# 		goal_root_dir,
# 		buffer_size,
# 		num_trans_context,
# 		in_mdp_batch_size,
# 	):
# 		trans_dataset = TransitionDataset(goal_root_dir=goal_root_dir, size=buffer_size)
# 		rand_sampler = RandomSampler(trans_dataset, replacement=False)

# 		self.data_loader = DataLoader(
# 			trans_dataset, 
# 			batch_size=num_trans_context + in_mdp_batch_size, 
# 			sampler=rand_sampler, 
# 			num_workers=1
# 		)
                        
# 		self.num_trans_context = num_trans_context
# 		self.in_mdp_batch_size = in_mdp_batch_size

# 	def sample(self, use_same_context=True):
		
# 		sampled_data = next(iter(self.data_loader))['trans']
# 		context_obs, _, context_actions, context_rewards, _ = [item[:self.num_trans_context] for item in sampled_data]

# 		obs, next_obs, actions, rewards, _ = [item[self.num_trans_context:] for item in sampled_data]

# 		context = np.concatenate([context_obs[None], context_actions[None], context_rewards[None]], axis=2)
# 		context = from_numpy(context)
		
# 		# Construct returning batch using the the sampled data
# 		batch = [obs, next_obs, actions, rewards, context]

# 		return batch


# @ray.remote(num_cpus=1.0)
# class MultiTaskReplayBuffer_dataLoader(object):
# 	def __init__(
# 			self,
# 			bcq_buffers_list,
# 		):
# 		torch.set_num_threads(1)
# 		self.num_tasks = len(bcq_buffers_list)
# 		self.task_buffers = dict([(idx, buffer) for idx, buffer in enumerate(bcq_buffers_list)])

# 	def sample_training_data(self, batch_idxes, use_same_context):

# 		batch_obj_ids = [
# 			self.task_buffers[idx].sample.remote(use_same_context) for i, idx in enumerate(batch_idxes)
# 		]
# 		return ray.get(batch_obj_ids)

# 	def end_epoch(self, epoch):
# 		pass
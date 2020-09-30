import numpy as np
import torch
import os

from utils.rng import set_seed

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
		use_next_obs_in_context=False,
	):
		torch.set_num_threads(1)
		self.storage = []
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
			
		ind = np.random.randint(
			0, self.buffer_size, size=(self.num_trans_context, )
		)
		sampled_data = self.storage[ind]
		obs = np.stack(list(sampled_data[:, 0]))
		next_obs = np.stack(list(sampled_data[:, 1]))
		actions = np.stack(list(sampled_data[:, 2]))
		rewards = sampled_data[:, 3].astype(np.float64).reshape(-1, 1)
	
		if self.use_next_obs_in_context:
			context = np.concatenate([obs, actions, next_obs, rewards], axis=1)
		else:
			context = np.concatenate([obs, actions, rewards], axis=1)

		# Sample training transitions
		sample_range = np.delete(np.arange(self.buffer_size), ind)
		ind = np.random.choice(
			sample_range, size=(self.in_mdp_batch_size, )
		)
		sampled_data = self.storage[ind]

		# Construct returning batch using the the sampled data
		batch = []

		batch.append(np.stack(list(sampled_data[:, 0]))) #obs
		batch.append(np.stack(list(sampled_data[:, 2]))) #actions

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

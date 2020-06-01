import numpy as np
import torch

import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy, get_numpy
from utils.rng import set_seed
from utils.pythonplusplus import load_gzip_pickle

import ray
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
	):
		torch.set_num_threads(1)
		self.storage = []
		self.num_trans_context = num_trans_context
		self.in_mdp_batch_size = in_mdp_batch_size
		self.buffer_size = 0
		set_seed(10 * seed + 2337 + index)

	# Expects tuples of (state, next_state, action, reward, done, vel_dif)
	def add(self, data):
		self.storage.append(data)

	def sample(self, use_same_context):
		# Sample context for each task
		if use_same_context:
			context_sample_size = self.num_trans_context
		else:
			assert False
			context_sample_size = self.num_trans_context * self.in_mdp_batch_size
			
		ind = np.random.randint(
			0, self.buffer_size, size=(context_sample_size, )
		)
		sampled_data = self.storage[ind]
		obs = np.stack(list(sampled_data[:, 0]))
		next_obs = np.stack(list(sampled_data[:, 1]))
		actions = np.stack(list(sampled_data[:, 2]))
		rewards = sampled_data[:, 3].astype(np.float64).reshape(-1, 1)
		
		context = np.concatenate([obs[None], actions[None], rewards[None], next_obs[None]], axis=2)

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
		batch.append(sampled_data[:, 4].astype(np.float64).reshape(-1, 1))

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

	def sample_training_data(self, batch_idxes, use_same_context):

		batch_obj_ids = [
			self.task_buffers[idx].sample.remote(use_same_context) for i, idx in enumerate(batch_idxes)
		]
		return batch_obj_ids

	def end_epoch(self, epoch):
		pass
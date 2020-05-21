import abc
from collections import OrderedDict, deque
import time

import gtimer as gt
from tqdm import trange
import numpy as np
import torch

from utils.logging import logger
import utils.pytorch_util  as ptu
import ray


class BatchMetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            train_buffer,
            train_goal_id,
            num_tasks,
            num_epochs,
            num_train_loops_per_epoch,
    ):
        super().__init__()

        self.train_buffer = train_buffer
        self.train_goal_id = train_goal_id
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.num_train_loops_per_epoch = num_train_loops_per_epoch

        self.trainer = trainer

    def train(self, start_epoch=0):

        batch_idxes = np.arange(self.num_tasks)
        batch_idxes = np.concatenate([batch_idxes[self.train_goal_id:], batch_idxes[:self.train_goal_id]])

        for epoch in gt.timed_for(
                trange(start_epoch, self.num_epochs),
                save_itrs=True,
        ):

            # Sample meta training tasks. And transfer the
            # transitions sampling job to each remote replay buffer.

            train_batch_obj_id = self.train_buffer.sample_training_data(batch_idxes)

            for _ in range(self.num_train_loops_per_epoch):
                train_raw_batch = ray.get(train_batch_obj_id)
                gt.stamp('sample_training_data', unique=False)

                # In this way, we can start the data sampling job for the
                # next training while doing training for the current loop.
                train_batch_obj_id = self.train_buffer.sample_training_data(batch_idxes)
                gt.stamp('set_up_sampling', unique=False)

                train_data = self.construct_training_batch(train_raw_batch)
                gt.stamp('construct_training_batch', unique=False)
                
                self.trainer.train(train_data, batch_idxes, epoch)
                
            gt.stamp('training', unique=False)

            self._end_epoch(epoch)

    def construct_training_batch(self, raw_batch):
        ''' Construct training batch from raw batch'''
        obs = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[0]) for rb in raw_batch
        ), dim=0)
        next_obs = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[1]) for rb in raw_batch
        ), dim=0)
        actions = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[2]) for rb in raw_batch
        ), dim=0)
        rewards = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[3]) for rb in raw_batch
        ), dim=0)
        qpos = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[4]) for rb in raw_batch
        ), dim=0)
        qvel = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[5]) for rb in raw_batch
        ), dim=0)
        return {
            'obs': obs,
            'next_obs': next_obs,
            'actions': actions,
            'rewards': rewards,
            'qpos': qpos,
            'qvel': qvel,
        }
        
    def _end_epoch(self, epoch):

        self._log_stats(epoch)
        if epoch > 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch + 1, snapshot)
        gt.stamp('saving', unique=False)

        self.trainer.end_epoch(epoch)

        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)

        write_header = True if epoch == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)

    def _get_snapshot(self, epoch):
        ''''
        Currently we do not need to get snapt shot
        '''
        snapshot = dict(
            trainer=self.trainer.get_snapshot(),
        )

        # What epoch indicates is that at the end of this epoch,
        # The state of the program is snapshot
        # Not to be confused with at the beginning of the epoch
        snapshot['epoch'] = epoch

        # Save the state of various rng
        snapshot['global_pkg_rng_state'] = get_global_pkg_rng_state()

        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        """
        Misc
        """
        gt.stamp('logging')

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times

def get_global_pkg_rng_state():

    rng = dict()

    rng['np_rng_state'] = np.random.get_state()
    rng['t_cpu_rng_state'] = torch.get_rng_state()

    if torch.cuda.is_available():
        rng['t_gpu_rng_state'] = torch.cuda.get_rng_state_all()

    return rng
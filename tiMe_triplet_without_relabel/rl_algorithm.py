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
            path_collector,
            train_buffer,
            train_goals,
            wd_goals,
            ood_goals,
            num_epochs,
            num_train_loops_per_epoch,
            num_tasks,
            num_workers,
    ):
        super().__init__()

        self.train_buffer = train_buffer
        self.num_epochs = num_epochs
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.path_collector = path_collector
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.train_goals = [goal for goal in train_goals]
        self.wd_goals = [goal for goal in wd_goals]
        self.ood_goals = [goal for goal in ood_goals]

        self.trainer = trainer

        self.avg_train_episode_returns = []
        self.final_train_achieved = []
        self.train_avg_returns = 0.

        self.avg_wd_episode_returns = []
        self.final_wd_achieved = []
        self.wd_avg_returns = 0.

        self.avg_ood_episode_returns = []
        self.final_ood_achieved = []
        self.ood_avg_returns = 0.

        self._start_epoch = 0

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):

        batch_idxes = np.arange(self.num_tasks)

        gt.start()

        for epoch in gt.timed_for(
                trange(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):  
            # Distribute the evaluation. We ship the 
            # params of each needed network to the 
            # remote path collector
            params_list = []
            for net in self.trainer.networks:
                params_list.append(ptu.state_dict_cpu(net))

            self.path_collector.set_network_params(params_list)

            gt.stamp('ship_params_to_evaluate', unique=False)

            evaluation_train_obj_id_list = []
            count = 0
            while count < len(self.train_goals) :
                if len(self.train_goals) - count < self.num_workers:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.train_goals[count:])
                    count = len(self.train_goals)
                else:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.train_goals[count:count + self.num_workers])
                    count += self.num_workers
                evaluation_train_obj_id_list.extend(evaluation_obj_id)

            assert len(evaluation_train_obj_id_list) == len(self.train_goals)

            evaluation_wd_obj_id_list = []
            count = 0
            while count < len(self.wd_goals) :
                if len(self.wd_goals) - count < self.num_workers:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.wd_goals[count:])
                    count = len(self.wd_goals)
                else:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.wd_goals[count:count + self.num_workers])
                    count += self.num_workers
                evaluation_wd_obj_id_list.extend(evaluation_obj_id)

            assert len(evaluation_wd_obj_id_list) == len(self.wd_goals)

            evaluation_ood_obj_id_list = []
            count = 0
            while count < len(self.ood_goals) :
                if len(self.ood_goals) - count < self.num_workers:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.ood_goals[count:])
                    count = len(self.ood_goals)
                else:
                    evaluation_obj_id = self.path_collector.async_evaluate(self.ood_goals[count:count + self.num_workers])
                    count += self.num_workers
                evaluation_ood_obj_id_list.extend(evaluation_obj_id)

            assert len(evaluation_ood_obj_id_list) == len(self.ood_goals)

            gt.stamp('set_up_evaluation', unique=False)

            # Sample meta training tasks. And transfer the
            # transitions sampling job to each remote replay buffer.
            train_batch_obj_id = self.train_buffer.sample_training_data(batch_idxes)

            for it in range(self.num_train_loops_per_epoch):
                train_raw_batch = ray.get(train_batch_obj_id)
                gt.stamp('sample_training_data', unique=False)

                # In this way, we can start the data sampling job for the
                # next training while doing training for the current loop.
                train_batch_obj_id = self.train_buffer.sample_training_data(batch_idxes)
                gt.stamp('set_up_sampling', unique=False)

                train_data = self.construct_training_batch(train_raw_batch)
                gt.stamp('construct_training_batch', unique=False)

                self.trainer.train(train_data, batch_idxes)

                if (it + 1) % 100 == 0:
                    print(it)

            eval_train_returns = ray.get(evaluation_train_obj_id_list)

            self.avg_train_episode_returns = [item[0] for item in eval_train_returns]
            self.final_train_achieved = [item[1] for item in eval_train_returns]
            self.train_avg_returns = np.mean(self.avg_train_episode_returns)
            
            eval_wd_returns = ray.get(evaluation_wd_obj_id_list)

            self.avg_wd_episode_returns = [item[0] for item in eval_wd_returns]
            self.final_wd_achieved = [item[1] for item in eval_wd_returns]
            self.wd_avg_returns = np.mean(self.avg_wd_episode_returns)

            eval_ood_returns = ray.get(evaluation_ood_obj_id_list)

            self.avg_ood_episode_returns = [item[0] for item in eval_ood_returns]
            self.final_ood_achieved = [item[1] for item in eval_ood_returns]
            self.ood_avg_returns = np.mean(self.avg_ood_episode_returns)

            gt.stamp('evaluation', unique=False)

            self._end_epoch(epoch)

    def construct_training_batch(self, raw_batch):
        ''' Construct training batch from raw batch'''
        # obs = np.concatenate([rb[0] for rb in raw_batch], axis=0)
        # actions = np.concatenate([rb[2] for rb in raw_batch], axis=0)
        # contexts = np.concatenate([rb[4] for rb in raw_batch], axis=0)
        # rewards = np.concatenate([rb[3] for rb in raw_batch], axis=0)
        # next_obs = np.concatenate([rb[1] for rb in raw_batch], axis=0)

        obs = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[0]) for rb in raw_batch
        ), dim=0)
        actions = torch.cat(tuple(
            ptu.elem_or_tuple_to_variable(rb[2]) for rb in raw_batch
        ), dim=0)

        # contexts: list of contexts from each tasks: 
        # (num_candidate_context, num_trans_context, context_dim)
        contexts = [ptu.elem_or_tuple_to_variable(rb[4]) for rb in raw_batch]

        return {
            'obs': obs,
            'actions': actions,
            'contexts': contexts,
        }
        
    def get_evaluation_diagnostics(self):
        eval = OrderedDict()

        eval['avg_train_episode_returns'] = self.avg_train_episode_returns
        eval['final_train_achieved'] = self.final_train_achieved
        eval['train_avg_returns'] = self.train_avg_returns

        eval['avg_wd_episode_returns'] = self.avg_wd_episode_returns
        eval['final_wd_achieved'] = self.final_wd_achieved
        eval['wd_avg_returns'] = self.wd_avg_returns

        eval['avg_ood_episode_returns'] = self.avg_ood_episode_returns
        eval['final_ood_achieved'] = self.final_ood_achieved
        eval['ood_avg_returns'] = self.ood_avg_returns
        return eval

    def _end_epoch(self, epoch):

        self._log_stats(epoch)
        if epoch > 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch + 1, snapshot)
        gt.stamp('saving', unique=False)

        self.trainer.end_epoch(epoch)
        self.path_collector.end_epoch(epoch)
        self.train_buffer.end_epoch(epoch)

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
        Evaluation
        """
        logger.record_dict(self.get_evaluation_diagnostics(), prefix='eval/')
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
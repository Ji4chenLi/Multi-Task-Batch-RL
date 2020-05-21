import gtimer as gt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import os.path as osp

from collections import OrderedDict, deque

import utils.pytorch_util as ptu
from utils.pytorch_util import np_to_pytorch_batch


class SuperQTrainer(object):
    def __init__(
            self,
            env,
            network_ensemble,
            train_goal,
            std_threshold,
            general_lr=3e-4,
            optimizer_class=optim.Adam,
    ):
        super().__init__()

        self.env = env
        self.env.set_goal(train_goal)
        self.env.reset()

        self.network_ensemble = network_ensemble
        self.std_threshold = std_threshold

        self.network_ensemble_optimizer = optimizer_class(
            [{'params': net.parameters()} for net in self.network_ensemble], lr=general_lr
        )

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train(self, batch, batch_idxes, epoch):
        """
        Unpack data from the batch
        """
        obs = batch['obs']
        actions = batch['actions']
        next_obs = batch['next_obs']
        qpos = batch['qpos']
        qvel = batch['qvel']
        
        # Get the in_mdp_batch_size
        in_mdp_batch_size = obs.shape[0] // batch_idxes.shape[0]
        num_tasks = batch_idxes.shape[0]

        """
        Obtain the model prediction loss
        """
        # Note that here, we do not calculate the obs_loss.

        next_obs_loss_task_0 = []
        pred_next_obs = [net(obs, actions) for net in self.network_ensemble]

        for pred_no in pred_next_obs:

            loss = F.mse_loss(pred_no[:in_mdp_batch_size], next_obs[:in_mdp_batch_size])
            next_obs_loss_task_0.append(loss)
 
        next_obs_magnitude = torch.mean(torch.norm(next_obs[:in_mdp_batch_size], dim=1))

        gt.stamp('get_tranistion_prediction_loss', unique=False)

        self.network_ensemble_optimizer.zero_grad()

        next_obs_loss_task_0 = torch.stack(next_obs_loss_task_0)
        next_obs_loss_task_0 = torch.sum(next_obs_loss_task_0)
        next_obs_loss_task_0.backward()

        # [loss.backward() for loss in next_obs_loss_task_0]

        self.network_ensemble_optimizer.step()

        gt.stamp('update', unique=False)

        """
        Save some statistics for eval
        """

        if self._need_to_update_eval_statistics:
            
            if epoch > 150:

                qpos_other_tasks = [qpos[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(0, batch_idxes.shape[0])]
                qvel_other_tasks = [qvel[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(0, batch_idxes.shape[0])]
                actions_other_tasks = [actions[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(0, batch_idxes.shape[0])]

                pred_next_obs_other_tasks = [torch.cat([

                    pred_no[in_mdp_batch_size*i : in_mdp_batch_size*(i + 1)][...,None] for pred_no in pred_next_obs

                ], dim=-1) for i in range(0, batch_idxes.shape[0])]

                next_obs_loss_other_tasks = []
                next_obs_loss_other_tasks_std = []
                num_selected_trans_other_tasks = []
                for item in zip(pred_next_obs_other_tasks, qpos_other_tasks, qvel_other_tasks, actions_other_tasks):
                    
                    pred_no_other_task, qp_other_task, qv_other_task, a_other_task = item

                    pred_std = torch.std(pred_no_other_task, dim=-1)
                    pred_std = pred_std.squeeze()
                    pred_std = torch.mean(pred_std, dim=1)

                    mask = ptu.get_numpy(pred_std < self.std_threshold)
                    num_selected_trans_other_tasks.append(np.sum(mask))

                    mask = mask.astype(bool)
                    pred_no_other_task = ptu.get_numpy(pred_no_other_task)
                    pred_no_other_task = pred_no_other_task[mask]

                    qp_other_task = ptu.get_numpy(qp_other_task)
                    qp_other_task = qp_other_task[mask]

                    qv_other_task = ptu.get_numpy(qv_other_task)
                    qv_other_task = qv_other_task[mask]

                    a_other_task = ptu.get_numpy(a_other_task)
                    a_other_task = a_other_task[mask]

                    mse_loss = []
                    for pred_no, qp, qv, a in zip(pred_no_other_task, qp_other_task, qv_other_task, a_other_task):
                        self.env.set_state(qp, qv)
                        no, _, _, _ = self.env.step(a)

                        loss = (pred_no - no.reshape(-1, 1)) ** 2
                        loss = np.mean(loss, axis=0)
                        mse_loss.append(loss)

                    if len(mse_loss) > 0:
                        mse_loss = np.stack(mse_loss)
                        mse_loss_mean = np.mean(mse_loss)
                        next_obs_loss_other_tasks.append(mse_loss_mean)

                        mse_loss_std = np.std(mse_loss, axis=1)
                        mse_loss_std = np.mean(mse_loss_std)
                        next_obs_loss_other_tasks_std.append(mse_loss_std)

                self.eval_statistics['average_task_next_obs_loss_other_tasks_mean'] = next_obs_loss_other_tasks
                self.eval_statistics['average_task_next_obs_loss_other_tasks_std'] = next_obs_loss_other_tasks_std

                self.eval_statistics['num_selected_trans_other_tasks'] = num_selected_trans_other_tasks

            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            # self.eval_statistics['next_obs_loss_task_0'] = np.mean(
            #     ptu.get_numpy(torch.mean(torch.stack(next_obs_loss_task_0)))
            # )
            self.eval_statistics['next_obs_loss_task_0'] = np.mean(
                ptu.get_numpy(next_obs_loss_task_0 / len(self.network_ensemble))
            )
            self.eval_statistics['next_obs_magnitude'] = ptu.get_numpy(next_obs_magnitude) 

            
            
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [net for net in self.network_ensemble]

    def get_snapshot(self):
        return dict(
            network_ensemble_state_dict=[net.state_dict() for net in self.network_ensemble],
            P_optimizer_state_dict=self.network_ensemble_optimizer.state_dict(),

            eval_statistics=self.eval_statistics,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
        )

    def restore_from_snapshot(self, ss):
        ss = ss['trainer']
        self.network_ensemble.load_state_dict(ss['network_ensemble_state_dict'])
        self.network_ensemble.to(ptu.device)

        self.eval_statistics = ss['eval_statistics']
        self._need_to_update_eval_statistics = ss['_need_to_update_eval_statistics']


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
            domain,
            general_lr=3e-4,
            optimizer_class=optim.Adam,
    ):
        super().__init__()

        self.env = env
        try:
            self.env.set_goal(train_goal)
        except AttributeError:
            self.env.set_target(train_goal)
        self.env.reset()

        self.network_ensemble = network_ensemble
        self.std_threshold = std_threshold

        self.network_ensemble_optimizer = optimizer_class(
            [{'params': net.parameters()} for net in self.network_ensemble], lr=general_lr
        )

        self.domain = domain
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train(self, batch, batch_idxes, epoch):
        """
        Unpack data from the batch
        """
        rewards = batch['rewards']
        obs = batch['obs']
        actions = batch['actions']

        # Get the in_mdp_batch_size
        in_mdp_batch_size = obs.shape[0] // batch_idxes.shape[0]
        """
        Obtain the model prediction loss
        """
        # Note that here, we do not calculate the obs_loss.

        pred_rewards = [net(obs, actions) for net in self.network_ensemble]

        # If you would like to train the reward estimator without
        # using the ensemble, please comment out Line 62 and uncomment
        # the Line 67 to train only one network to predict the rewards
        # pred_rewards = [self.network_ensemble[0](obs, actions) for net in self.network_ensemble]

        reward_loss_task_0 = [F.mse_loss(pred_r[:in_mdp_batch_size], rewards[:in_mdp_batch_size]) for pred_r in pred_rewards]
        gt.stamp('get_reward_loss', unique=False)

        self.network_ensemble_optimizer.zero_grad()

        [loss.backward() for loss in reward_loss_task_0]

        # Please comment out Line 74 and uncomment Line 78 if you would
        # like to train the reward estimator without using the ensemble
        # reward_loss_task_0[0].backward()

        self.network_ensemble_optimizer.step()

        gt.stamp('update', unique=False)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            
            if epoch > -1:
                obs_other_tasks = [obs[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(0, batch_idxes.shape[0])]
                actions_other_tasks = [actions[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(0, batch_idxes.shape[0])]
                pred_rewards_other_tasks = [torch.cat(
                    [pred_r[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for pred_r in pred_rewards
                ], dim=1) for i in range(0, batch_idxes.shape[0])]

                reward_loss_other_tasks = []
                reward_loss_other_tasks_std = []
                reward_loss_prop_other_tasks = []
                num_selected_trans_other_tasks = []
                for i, item in enumerate(zip(pred_rewards_other_tasks, obs_other_tasks, actions_other_tasks)):
                    pred_r_other_task, o_other_task, a_other_task = item
                    pred_std = torch.std(pred_r_other_task, dim=1)
                    # print(pred_std)
                    mask = ptu.get_numpy(pred_std < self.std_threshold)
                    num_selected_trans_other_tasks.append(np.sum(mask))

                    mask = mask.astype(bool)
                    pred_r_other_task = ptu.get_numpy(pred_r_other_task)
                    pred_r_record = pred_r_other_task[mask]
                    o_other_task = ptu.get_numpy(o_other_task)
                    o_other_task = o_other_task[mask]
                    a_other_task = ptu.get_numpy(a_other_task)
                    a_other_task = a_other_task[mask]

                    mse_loss = []
                    mse_loss_prop = []

                    for pred_r, o, a in zip(pred_r_record, o_other_task, a_other_task):
                        if self.domain == 'ant-dir':
                            qpos = np.concatenate([np.zeros(2), o[:13]])
                            qvel = o[13:27]
                        elif self.domain == 'ant-goal':
                            qpos = o[:15]
                            qvel = o[15:29]
                        elif self.domain == 'humanoid-ndone-goal':
                            qpos = o[:24]
                            qvel = o[24:47]
                        elif self.domain == 'humanoid-openai-dir':
                            qpos = np.concatenate([np.zeros(2), o[:22]])
                            qvel = o[22:45]
                        elif self.domain == 'halfcheetah-vel':
                            qpos = np.concatenate([np.zeros(1), o[:8]])
                            qvel = o[8:17]
                        elif 'maze' in self.domain:
                            qpos = o[:2]
                            qvel = o[2:4]

                        self.env.set_state(qpos, qvel)
                        _, r, _, _ = self.env.step(a)
                        mse_loss.append((pred_r - r) ** 2)
                        mse_loss_prop.append(np.sqrt((pred_r - r) ** 2 / r ** 2))

                    if len(mse_loss) > 0:
                        reward_loss_other_tasks.append(np.mean(np.stack(mse_loss), axis=0).tolist())
                        reward_loss_other_tasks_std.append(np.std(np.stack(mse_loss), axis=0).tolist())
                        reward_loss_prop_other_tasks.append(np.mean(np.stack(mse_loss_prop), axis=0).tolist())

                self.eval_statistics['average_task_reward_loss_other_tasks_mean'] = np.mean(reward_loss_other_tasks, axis=1)
                self.eval_statistics['average_task_reward_loss_other_tasks_std'] = np.std(reward_loss_other_tasks, axis=1)
                self.eval_statistics['average_task_reward_loss_prop_other_task'] = np.mean(reward_loss_prop_other_tasks, axis=1)

                self.eval_statistics['num_selected_trans_other_tasks'] = num_selected_trans_other_tasks

            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['reward_loss_task_0'] = np.mean(
                ptu.get_numpy(torch.mean(torch.stack(reward_loss_task_0)))
            )
            
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


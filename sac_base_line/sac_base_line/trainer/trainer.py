from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import np_to_pytorch_batch

import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable


class SACTrainer(object):
    def __init__(
            self,
            policy_producer,
            qf1, target_qf1,
            qf2, target_qf2,
            lr,

            action_space=None,

            discount=0.99,
            reward_scale=1.0,

            optimizer_class=optim.Adam,

            soft_target_tau_qf=5e-3,
            soft_target_tau_policy=1e-2,
            target_update_period=1,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = - \
                    np.prod(action_space.shape).item()

        self.soft_target_tau_qf = soft_target_tau_qf
        self.soft_target_tau_policy = soft_target_tau_policy

        self.target_update_period = target_update_period

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale

        """
        The class mutable state
        """

        self.policy = policy_producer()
        self.target_policy = policy_producer()
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        if self.use_automatic_entropy_tuning:
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=3e-4,
            )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )

        self.policy_imitation_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=3e-4,
        )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=lr,
        )
        
        print('----------------------------------')
        print('qf_optimizer learning rate: ', lr)
        print('soft_target_tau_qf: ', soft_target_tau_qf)
        print('soft_target_tau_policy: ', soft_target_tau_policy)
        print('----------------------------------')
        
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_qf1(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch_qf1(batch)

    def train_from_torch_qf1(self, batch):

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.use_automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = 1

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)

        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q1_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q1_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q1_values

        qf1_loss = self.qf_criterion(q1_pred, q1_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau_qf
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Targets',
                ptu.get_numpy(q1_target),
            ))

        self._n_train_steps_total += 1

    def train_qf2_policy(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch_qf2_policy(batch)

    def train_from_torch_qf2_policy(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        QF Loss
        """
        if self.use_automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = 1

        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q2_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q2_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q2_values
        qf2_loss = self.qf_criterion(q2_pred, q2_target.detach())

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi +
                            self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        Update networks
        """
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau_qf
            )

            ptu.soft_update_from_to(
                self.policy, self.target_policy, self.soft_target_tau_policy
            )
            self.policy.load_state_dict(self.target_policy.state_dict())

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Targets',
                ptu.get_numpy(q2_target),
            ))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        return [
            self.policy,
            self.target_policy,
            self.qf1,
            self.target_qf1,
            self.qf2,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),

            qf1_state_dict=self.qf1.state_dict(),
            qf1_optim_state_dict=self.qf1_optimizer.state_dict(),
            target_qf1_state_dict=self.target_qf1.state_dict(),

            qf2_state_dict=self.qf2.state_dict(),
            qf2_optim_state_dict=self.qf2_optimizer.state_dict(),
            target_qf2_state_dict=self.target_qf2.state_dict(),

            log_alpha=self.log_alpha,
            alpha_optim_state_dict=self.alpha_optimizer.state_dict(),

            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
        )

    # def restore_from_snapshot(self, ss):

    #     policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

    #     self.policy.load_state_dict(policy_state_dict)
    #     self.policy_optimizer.load_state_dict(policy_optim_state_dict)

    #     qf1_state_dict, qf1_optim_state_dict = ss['qf1_state_dict'], ss['qf1_optim_state_dict']
    #     target_qf1_state_dict = ss['target_qf1_state_dict']

    #     self.qf1.load_state_dict(qf1_state_dict)
    #     self.qf1_optimizer.load_state_dict(qf1_optim_state_dict)
    #     self.target_qf1.load_state_dict(target_qf1_state_dict)

    #     qf2_state_dict, qf2_optim_state_dict = ss['qf2_state_dict'], ss['qf2_optim_state_dict']
    #     target_qf2_state_dict = ss['target_qf2_state_dict']

    #     self.qf2.load_state_dict(qf2_state_dict)
    #     self.qf2_optimizer.load_state_dict(qf2_optim_state_dict)
    #     self.target_qf2.load_state_dict(target_qf2_state_dict)

    #     log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']

    #     self.log_alpha.data.copy_(log_alpha)
    #     self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

    #     self.eval_statistics = ss['eval_statistics']
    #     self._n_train_steps_total = ss['_n_train_steps_total']
    #     self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']

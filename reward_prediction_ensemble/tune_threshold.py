import torch
import numpy as np
import pickle
import utils.pytorch_util as ptu
from utils.env_utils import env_producer
import os
import os.path as osp
from collections import OrderedDict
from utils.logging import logger, setup_logger
import datetime
import dateutil.tz


def create_simple_exp_name():
    """
    Create a unique experiment name with a timestamp
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return timestamp

std_threshold = 0.1
in_mdp_batch_size = 128

eval_statistics = OrderedDict()

logger.reset()
setup_logger(log_dir=osp.join('./tune_threshold_loggings', create_simple_exp_name()))

filename = f'./goals/ant-dir-normal-goals.pkl'
train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))

env = env_producer('ant-dir', 0, train_goals[0])

for epoch in range(200):

    file_name = osp.join('./data_reward_predictions', f'params_{epoch}.pkl')
    params = pickle.load(open(file_name, "rb"))

    obs = params['obs']
    actions = params['actions']
    rewards = params['rewards']
    pred_rewards = params['pred_rewards']

    obs_other_tasks = [obs[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(1, 32)]
    actions_other_tasks = [actions[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for i in range(1, 32)]
    pred_rewards_other_tasks = [np.concatenate(
        [pred_r[in_mdp_batch_size*i:in_mdp_batch_size*(i + 1)] for pred_r in pred_rewards
    ], axis=1) for i in range(1, 32)]

    reward_loss_other_tasks = []
    reward_loss_other_tasks_std = []
    num_selected_trans_other_tasks = []
    for i, item in enumerate(zip(pred_rewards_other_tasks, obs_other_tasks, actions_other_tasks)):
        # env.set_goal(train_goals[i + 1])
        pred_r_other_task, o_other_task, a_other_task = item
        pred_std = np.std(pred_r_other_task, axis=1)
        mask = pred_std < std_threshold
        num_selected_trans_other_tasks.append(np.sum(mask))

        mask = mask.astype(bool)
        pred_r_record = pred_r_other_task[mask]
        o_other_task = o_other_task[mask]
        a_other_task = a_other_task[mask]

        mse_loss = []
        for pred_r, o, a in zip(pred_r_record, o_other_task, a_other_task):
            qpos = np.concatenate([np.zeros(2), o[:13]])
            qvel = o[13:27]
            env.set_state(qpos, qvel)
            _, r, _, _ = env.step(a)
            mse_loss.append((pred_r - r) ** 2)
        reward_loss_other_tasks.append(np.mean(np.stack(mse_loss), axis=0).tolist())
        reward_loss_other_tasks_std.append(np.std(np.stack(mse_loss), axis=0).tolist())

    eval_statistics['reward_loss_other_tasks'] = reward_loss_other_tasks
    eval_statistics['reward_loss_other_tasks_std'] = reward_loss_other_tasks_std
    eval_statistics['average_ensemble_reward_loss_other_tasks_mean'] = np.mean(reward_loss_other_tasks, axis=0)
    eval_statistics['average_ensemble_reward_loss_other_tasks_std'] = np.std(reward_loss_other_tasks, axis=0)

    eval_statistics['average_task_reward_loss_other_tasks_mean'] = np.mean(reward_loss_other_tasks, axis=1)
    eval_statistics['average_task_reward_loss_other_tasks_std'] = np.std(reward_loss_other_tasks, axis=1)

    eval_statistics['num_selected_trans_other_tasks'] = num_selected_trans_other_tasks

    logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
    logger.record_dict(eval_statistics, prefix='trainer/')

    write_header = True if epoch == 0 else False
    logger.dump_tabular(with_prefix=False, with_timestamp=False,
                        write_header=write_header)

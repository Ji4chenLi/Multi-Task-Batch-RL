import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_


class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()


class HalfCheetahVelEnv(HalfCheetahEnv):

    def __init__(self, goal=None):
        if goal is None:
            goal = 1.0
        self._goal = goal
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     vel=forward_vel,
                     achieved=forward_vel)
        return (observation, reward, done, infos)

    def sample_goals(self, exp_mode):
        if exp_mode == 'interpolate':
            goals_left = np.random.uniform(0.0, 1.5, size=(16,))
            goals_right = np.random.uniform(2.5, 3.0, size=(16,))
            train_goals = np.concatenate((goals_left, goals_right))

            goals_left = np.random.uniform(0.0, 1.5, size=(16,))
            goals_right = np.random.uniform(2.5, 3.0, size=(16,))
            wd_goals = np.concatenate((goals_left, goals_right))

            ood_goals = np.random.uniform(2.5, 3.0, size=(8,))
        elif exp_mode == 'medium':
            train_goals = np.random.uniform(0.0, 2.5, size=(32,))
            wd_goals = np.random.uniform(0.0, 2.5, size=(8,))
            ood_goals = np.random.uniform(2.5, 3.0, size=(8,))
        elif exp_mode == 'hard':
            train_goals = np.random.uniform(0.0, 1.5, size=(32,))
            wd_goals = np.random.uniform(0.0, 1.5, size=(8,))
            ood_goals = np.random.uniform(2.5, 3.0, size=(8,))
        else:
            raise(NotImplementedError)

        return train_goals, wd_goals, ood_goals

    def set_goal(self, goal):
        self._goal = goal
        self.reset()

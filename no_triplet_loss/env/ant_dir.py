import numpy as np
from gym.envs.mujoco import AntEnv


class AntDirEnv(AntEnv):

    def __init__(self, goal=None):
        if goal is None:
            goal = 1.0
        self._goal = goal
        super(AntDirEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            achieved=torso_velocity[:2] / self.dt,
        )

    def sample_goals(self, exp_mode):
        if exp_mode == 'normal':
            train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi
            wd_goals = np.random.uniform(0., 1.5, size=(8,)) * np.pi
            ood_goals = np.random.uniform(1.5, 2.0, size=(8,)) * np.pi
        else:
            raise NotImplementedError

        return train_goals, wd_goals, ood_goals
    
    def set_goal(self, goal):
        self._goal = goal
        self.reset()

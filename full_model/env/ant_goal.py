import numpy as np
from gym.envs.mujoco import AntEnv as AntEnv_

class AntEnv(AntEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


class AntGoalEnv(AntEnv):
    def __init__(self, goal=None):
        if goal is None:
            goal = self.sample_tasks(32)[0]['goal']
        self._goal = goal
        super(AntGoalEnv, self).__init__()
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            achieved=xposafter[:2],
        )

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def sample_goals(self, exp_mode):
        if exp_mode == 'normal':
            a = np.random.random(32) * 1.5 * np.pi
            r = 3 * np.random.random(32) ** 0.5
            train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

            a = np.random.random(8) * 1.5 * np.pi
            r = 3 * np.random.random(8) ** 0.5
            wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

            a = np.random.uniform(1.5, 2.0, size=(8,)) * np.pi
            r = 3 * np.random.random(8) ** 0.5
            ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        else:
            raise NotImplementedError

        return train_goals, wd_goals, ood_goals

    def set_goal(self, goal):
        self._goal = goal
        self.reset()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
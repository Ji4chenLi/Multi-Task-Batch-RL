import numpy as np

from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidGoalEnvOpenAI(HumanoidEnv):

    def __init__(self, goal=None):
        if goal is None:
            goal = np.zeros(2)
        self._goal = goal
        super(HumanoidGoalEnvOpenAI, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        goal_reward = -np.sum(np.abs(pos_after - self._goal))

        data = self.sim.data
        
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = goal_reward - quad_ctrl_cost - quad_impact_cost
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        return self._get_obs(), reward, done, dict(goal_reward=goal_reward,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_impact=-quad_impact_cost,
                                                   achieved=pos_after)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def set_goal(self, goal):
        self._goal = goal
        self.reset()

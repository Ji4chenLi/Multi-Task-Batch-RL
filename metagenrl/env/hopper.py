import numpy as np
from gym.envs.mujoco import HopperEnv as HopperEnv_


class HopperEnv(HopperEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()


class HopperVelEnv(HopperEnv):

    def __init__(self, goal_vel=None):
        self._goal_vel = goal_vel
        super(HopperVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 1e-3 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost + alive_bonus

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))

        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost)

        return (observation, reward, done, infos)

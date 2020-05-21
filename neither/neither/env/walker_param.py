import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils

class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0, goal=None):
        self._goal = 1.0
        RandomEnv.__init__(self, log_scale_limit, 'walker2d.xml', 5)
        utils.EzPickle.__init__(self)
        if goal is not None:
            self.set_task(goal)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, dict(achieved=((posafter - posbefore) / self.dt))

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def set_goal(self, goal):
        self.set_task(goal)
        self._goal = np.float64(-1.0)
        self.reset()

import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils

class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0, goal=None):
        self._goal = goal
        RandomEnv.__init__(self, log_scale_limit, 'walker2d.xml', 5)
        utils.EzPickle.__init__(self)

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

    def sample_tasks(self, n_tasks, is_train, is_within_distribution):
        if is_train:
            np.random.seed(1337)
        else:
            np.random.seed(2337)
        if is_within_distribution:
            param_sets = []
            for _ in range(n_tasks):
                # body mass -> one multiplier for all body parts

                new_params = {}

                if 'body_mass' in self.rand_params:
                    body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-1.5, 1.5,  size=self.model.body_mass.shape)
                    new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

                # body_inertia
                if 'body_inertia' in self.rand_params:
                    body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-1.5, 1.5,  size=self.model.body_inertia.shape)
                    new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

                # damping -> different multiplier for different dofs/joints
                if 'dof_damping' in self.rand_params:
                    dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-1.5, 1.5, size=self.model.dof_damping.shape)
                    new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

                # friction at the body components
                if 'geom_friction' in self.rand_params:
                    geom_friction_multipliers = np.array(1.5) ** np.random.uniform(-1.5, 1.5, size=self.model.geom_friction.shape)
                    new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], geom_friction_multipliers)

                param_sets.append(new_params)
        
        else:
            param_sets = []
            for _ in range(n_tasks):
                # body mass -> one multiplier for all body parts

                new_params = {}

                if 'body_mass' in self.rand_params:
                    key = np.random.uniform(0, 2)
                    if key < 1:
                        body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-3.0, -2.0,  size=self.model.body_mass.shape)
                    else:
                        body_mass_multiplyers = np.array(1.5) ** np.random.uniform(2.0, 3.0,  size=self.model.body_mass.shape)
                    new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

                # body_inertia
                if 'body_inertia' in self.rand_params:
                    key = np.random.uniform(0, 2)
                    if key < 1:
                        body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-3.0, -2.0,  size=self.model.body_inertia.shape)
                    else:
                        body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(2.0, 3.0,  size=self.model.body_inertia.shape)
                    new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

                # damping -> different multiplier for different dofs/joints
                if 'dof_damping' in self.rand_params:
                    key = np.random.uniform(0, 2)
                    if key < 1:
                        dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-3.0, -2.0, size=self.model.dof_damping.shape)
                    else:
                        dof_damping_multipliers = np.array(1.3) ** np.random.uniform(2.0, 3.0, size=self.model.dof_damping.shape)
                    new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

                # friction at the body components
                if 'geom_friction' in self.rand_params:
                    key = np.random.uniform(0, 2)
                    if key < 1:
                        geom_friction_multipliers = np.array(1.5) ** np.random.uniform(-3.0, -2.0, size=self.model.geom_friction.shape)
                    else:
                        geom_friction_multipliers = np.array(1.5) ** np.random.uniform(2.0, 3.0, size=self.model.geom_friction.shape)
                    new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], geom_friction_multipliers)

                param_sets.append(new_params)

        return param_sets


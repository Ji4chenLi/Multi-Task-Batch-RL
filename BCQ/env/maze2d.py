from pointmaze.maze_model import MazeEnv, U_MAZE, MEDIUM_MAZE
import numpy as np


class MazeUmazeEnv(MazeEnv):
    def __init__(
        self, 
        goal=None, 
        maze_spec=U_MAZE,
        reward_type='dense',
        reset_target=False
    ):
        super(MazeUmazeEnv, self).__init__(maze_spec=maze_spec, reward_type=reward_type, reset_target=False)
        self.set_target(goal)

    def reset_model(self):
        # Always initialize the agent at the middle of the connection 
        reset_location = np.array([2, 3]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs() 


class MazeMediumEnv(MazeEnv):
    def __init__(
        self, 
        goal=None, 
        maze_spec=MEDIUM_MAZE,
        reward_type='dense',
        reset_target=False
    ):
        super(MazeMediumEnv, self).__init__(maze_spec=maze_spec, reward_type=reward_type, reset_target=False)
        self.set_target(goal)

    def reset_model(self):
        # Always initialize the agent at the middle of the connection 
        reset_location = np.array([6, 1]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs() 

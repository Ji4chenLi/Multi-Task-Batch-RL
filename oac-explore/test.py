from pointmaze.maze_model import parse_maze, U_MAZE, MEDIUM_MAZE, MEDIUM_MAZE_EVAL
import numpy as np
import matplotlib.pyplot as plt
import pickle

WALL = 10
EMPTY = 11
GOAL = 12



maze_arr = parse_maze(U_MAZE)
print(maze_arr)

filename = f'./goals/maze-umaze-normal-goals.pkl'
_, train_goals, wd_goals, _ = pickle.load(open(filename, 'rb'))

maze_arr[1, 1] = 11
maze_arr = np.array(maze_arr)

plt.figure()
plt.imshow(maze_arr)
plt.scatter(np.array([3]), np.array([2]), label='Start')
plt.scatter(train_goals[:, 1], train_goals[:, 0], label='Train')
plt.scatter(wd_goals[:, 1], wd_goals[:, 0], label='WD Test')
plt.legend()
plt.savefig('umaze.png')
plt.show()

maze_arr = parse_maze(MEDIUM_MAZE)
print(maze_arr)

filename = f'./goals/maze-medium-normal-goals.pkl'
_, train_goals, wd_goals, _ = pickle.load(open(filename, 'rb'))

maze_arr[6, 6] = 11
maze_arr = np.array(maze_arr)

plt.figure()
plt.imshow(maze_arr)
plt.scatter(np.array([1]), np.array([6]), label='Start')
plt.scatter(train_goals[:, 1], train_goals[:, 0], label='Train')
plt.scatter(wd_goals[:, 1], wd_goals[:, 0], label='WD Test')
plt.legend()
plt.savefig('medium.png')
plt.show()


reset_locations = list(zip(*np.where(maze_arr == EMPTY)))
print(reset_locations)
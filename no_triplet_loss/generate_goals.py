import numpy as np
import pickle
from utils.env_utils import env_producer


#--------------------HalfCheetah-Vel-Hard----------------------

# np.random.seed(1337)

# train_goals = np.random.uniform(0.0, 1.5, size=(32,))
# wd_goals = np.random.uniform(0.5, 1.5, size=(8,))
# ood_goals = np.random.uniform(1.5, 2.5, size=(8,))

# idx_list = [3, 5, 8, 15, 16, 17, 23, 24, 29, 31]

# train_goals = train_goals[idx_list]

# filename = './goals/halfcheetah-vel-hard-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #-------------------Ant-Goal-Normal---------------------

# np.random.seed(1337)

# a = np.random.random(10) * np.pi * 2 / 3
# r = 3 * np.random.random(10) ** 0.5
# train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.random(8) * np.pi  * 2 / 3
# r = 3 * np.random.random(8) ** 0.5
# wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi
# r = 3 * np.random.random(8) ** 0.5
# ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# idx_list = list(range(10))
# train_goals = train_goals[idx_list]

# filename = './goals/ant-goal-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #-------------------Humanoid-Goal-Normal--------------------------

# np.random.seed(1337)

# a = np.random.random(10) * np.pi * 2 / 3
# r = 3 * np.random.random(10) ** 0.5
# train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.random(8) * np.pi  * 2 / 3
# r = 3 * np.random.random(8) ** 0.5
# wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi
# r = 3 * np.random.random(8) ** 0.5
# ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# idx_list = list(range(10))
# train_goals = train_goals[idx_list]

# filename = './goals/humanoid-goal-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

#-------------------Humanoid-OpenAI-Goal-Normal--------------------------

# np.random.seed(1337)

# a = np.random.random(10) * np.pi * 2 / 3
# r = 3 * np.random.random(10) ** 0.5
# train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.random(8) * np.pi  * 2 / 3
# r = 3 * np.random.random(8) ** 0.5
# wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# a = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi
# r = 3 * np.random.random(8) ** 0.5
# ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

# idx_list = list(range(10))
# train_goals = train_goals[idx_list]

# filename = './goals/humanoid-openai-goal-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #--------------------Ant-Dir-Normal---------------------------

# np.random.seed(1337)

# train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi

# wd_goals = np.random.uniform(0, 1, size=(8,)) * np.pi * 2 / 3
# ood_goals = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi

# idx_list = [0, 1, 4, 10, 12, 14, 17, 21, 26, 27]
# train_goals = train_goals[idx_list]

# filename = './goals/ant-dir-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #-------------------Humanoid-Dir-Normal--------------------------

# np.random.seed(1337)

# train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi

# wd_goals = np.random.uniform(0, 1, size=(8,)) * np.pi * 2 / 3
# ood_goals = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi

# idx_list = [0, 1, 4, 10, 12, 14, 17, 21, 26, 27]
# train_goals = train_goals[idx_list]

# filename = './goals/humanoid-dir-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #-------------------Humanoid-Dir-OpenAI-Normal--------------------------

# np.random.seed(1337)

# train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi

# wd_goals = np.random.uniform(0, 1, size=(8,)) * np.pi * 2 / 3
# ood_goals = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi

# idx_list = [0, 1, 4, 10, 12, 14, 17, 21, 26, 27]
# train_goals = train_goals[idx_list]

# filename = './goals/humanoid-openai-dir-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# print([idx_list, train_goals, wd_goals, ood_goals])

# #---------------------Walker-Param-Normal-------------------------

# sample_env = env_producer('walker-param', 0)
# train_goals = sample_env.sample_tasks(32, is_within_distribution=True)
# wd_goals = sample_env.sample_tasks(8, is_within_distribution=True)
# ood_goals = sample_env.sample_tasks(8, is_within_distribution=False)

# idx_list = list(range(30))
# train_goals = train_goals[:30]

# filename = './goals/walker-param-normal-goals.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

# # print([idx_list, train_goals, wd_goals, ood_goals])

#---------------------Maze-Umaze-Noraml-------------------------

np.random.seed(1336)

x_up = np.random.uniform(0.4, 1.2, size=(5,))
x_down = np.random.uniform(2.4, 3.2, size=(5,))
x = np.concatenate([x_up, x_down])
y = np.random.uniform(1.0, 2.0, size=(10,))
train_goals = np.stack([x, y], axis=-1)

x_up = np.random.uniform(0.4, 1.2, size=(5,))
x_down = np.random.uniform(2.4, 3.2, size=(5,))
x = np.concatenate([x_up, x_down])
y = np.random.uniform(1.0, 2.0, size=(10,))
wd_goals = np.stack([x, y], axis=-1)

ood_goals = np.random.uniform(0.5, 1.5, size=(8,))

idx_list = list(range(10))

filename = './goals/maze-umaze-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

print([idx_list, train_goals, wd_goals, ood_goals])

#-----------------------Maze-Medium-Normal------------------------------

np.random.seed(1336)

x_left = np.random.uniform(0.4, 2.2, size=(5,))
y_left = np.random.uniform(0.4, 2.2, size=(5,))
train_goals_left = np.stack([x_left, y_left], axis=-1)

x_right = np.random.uniform(0.4, 2.2, size=(5,))
y_right = np.random.uniform(4.4, 6.2, size=(5,))
train_goals_right = np.stack([x_right, y_right], axis=-1)

train_goals = np.concatenate([train_goals_left, train_goals_right], axis=0)

x_left = np.random.uniform(0.4, 2.2, size=(5,))
y_left = np.random.uniform(0.4, 2.2, size=(5,))
wd_goals_left = np.stack([x_left, y_left], axis=-1)

x_right = np.random.uniform(0.4, 2.2, size=(5,))
y_right = np.random.uniform(4.4, 6.2, size=(5,))
wd_goals_right = np.stack([x_right, y_right], axis=-1)

wd_goals = np.concatenate([wd_goals_left, wd_goals_right], axis=0)
ood_goals = np.random.uniform(0.5, 1.5, size=(8,))

idx_list = list(range(10))

filename = './goals/maze-medium-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

print([idx_list, train_goals, wd_goals, ood_goals])
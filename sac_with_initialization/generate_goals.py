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

np.random.seed(1337)

a = np.random.random(10) * np.pi * 2 / 3
r = 3 * np.random.random(10) ** 0.5
train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

a = np.random.random(8) * np.pi  * 2 / 3
r = 3 * np.random.random(8) ** 0.5
wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

a = np.random.uniform(2 / 3, 1.0, size=(8,)) * np.pi
r = 3 * np.random.random(8) ** 0.5
ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

idx_list = list(range(10))
train_goals = train_goals[idx_list]

filename = './goals/humanoid-openai-goal-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([idx_list, train_goals, wd_goals, ood_goals], f)

print([idx_list, train_goals, wd_goals, ood_goals])

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
# #----------------------------------------------------------------

import numpy as np
import pickle
from utils.env_utils import env_producer


np.random.seed(1337)

train_goals = np.random.uniform(0.0, 1.5, size=(32,))
wd_goals = np.random.uniform(0.0, 1.5, size=(8,))
ood_goals = np.random.uniform(2.5, 3.0, size=(8,))

filename = './goals/halfcheetah-vel-hard-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([train_goals, wd_goals, ood_goals], f)

print([train_goals, wd_goals, ood_goals])

np.random.seed(1337)

a = np.random.random(32) * 1.5 * np.pi
r = 3 * np.random.random(32) ** 0.5
train_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

a = np.random.random(8) * 1.5 * np.pi
r = 3 * np.random.random(8) ** 0.5
wd_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

a = np.random.uniform(1.5, 2.0, size=(8,)) * np.pi
r = 3 * np.random.random(8) ** 0.5
ood_goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

filename = './goals/ant-goal-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([train_goals, wd_goals, ood_goals], f)

print([train_goals, wd_goals, ood_goals])

np.random.seed(1337)

train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi
wd_goals = np.random.uniform(0., 1.5, size=(8,)) * np.pi
ood_goals = np.random.uniform(1.5, 2.0, size=(8,)) * np.pi

filename = './goals/ant-dir-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([train_goals, wd_goals, ood_goals], f)

print([train_goals, wd_goals, ood_goals])

np.random.seed(1337)

train_goals = np.random.uniform(0., 1.5, size=(32,)) * np.pi
wd_goals = np.random.uniform(0., 1.5, size=(8,)) * np.pi
ood_goals = np.random.uniform(1.5, 2.0, size=(8,)) * np.pi

filename = './goals/humanoid-dir-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([train_goals, wd_goals, ood_goals], f)

print([train_goals, wd_goals, ood_goals])

np.random.seed(1337)

sample_env = env_producer('walker-param', 0)
train_goals = sample_env.sample_tasks(32, is_within_distribution=True)
wd_goals = sample_env.sample_tasks(8, is_within_distribution=True)
ood_goals = sample_env.sample_tasks(8, is_within_distribution=False)

filename = './goals/walker-param-normal-goals.pkl'
with open(filename, 'wb') as f:
    pickle.dump([train_goals, wd_goals, ood_goals], f)

# print([train_goals, wd_goals, ood_goals])
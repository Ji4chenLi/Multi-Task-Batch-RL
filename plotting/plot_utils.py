import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import ast
import copy


domain_name = []

domain_to_title = {
    'ant-dir': 'AntDir', 
    'ant-goal': 'AntGoal', 
    'humanoid-openai-dir': 'HumanoidDir-M',  
    'halfcheetah-vel': 'HalfCheetahVel', 
    'walker-param': 'WalkerParam',
    'maze-umaze': 'UmazeGoal-M',
}

domain_to_epoch = {
    'halfcheetah-vel': 50,
    'ant-goal': 590,
    'ant-dir': 590,
    'humanoid-openai-dir':590,
    'walker-param': 390,
}


def smooth_results(results, smoothing_window=30):
    # Codes adapted from the "https://github.com/microsoft/oac-explore"
    smoothed = np.zeros((results.shape[0], results.shape[1]))

    for idx in range(len(smoothed)):

        if idx == 0:
            smoothed[idx] = results[idx]
            continue

        start_idx = max(0, idx - smoothing_window)

        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)

    return smoothed


def plot(values, label, color=[0, 0, 1, 1]):
    # Codes adapted from the "https://github.com/microsoft/oac-explore"
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)

    x_vals = np.arange(len(mean))

    blur = copy.deepcopy(color)
    blur[-1] = 0.1

    plt.plot(x_vals, mean, label=label, color=color)
    plt.fill_between(x_vals, mean - std, mean + std, color=blur)


def get_results(time, bcq_encoder=False):
    wd_returns = time['eval/avg_wd_episode_returns'].to_list()
    wd_return_list = []

    for i in range(len(wd_returns)):
        wd_re = np.array(ast.literal_eval(wd_returns[i]))
        wd_return_list.append(wd_re)
        
    wd_return_smooth = smooth_results(np.array(wd_return_list))

    return wd_return_smooth

def get_pearl_results(pearl):
    wd_returns = pearl['Return_all_wd_tasks'].to_list()
    wd_return_list = []

    for i in range(len(wd_returns)):
        wd_re = np.array(ast.literal_eval(wd_returns[i]))
        wd_return_list.append(wd_re)

    wd_return_smooth_pearl = smooth_results(np.array(wd_return_list))
    
    return wd_return_smooth_pearl
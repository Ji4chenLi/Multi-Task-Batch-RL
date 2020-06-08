import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import importlib
import os
import pickle

from plotting.plot_utils import plot, domain_to_title, domain_to_epoch


if __name__ == "__main__":

    domain = 'humanoid-openai-dir'
    variant_module = importlib.import_module(f'sac_with_initialization.configs.{domain}')
    variant = variant_module.params

    seed = 0
    exp_mode = variant['exp_mode']
    max_path_length = variant['max_path_length']

    with_ensemble_returns = []
    conven_returns = []
    without_ensemble_returns = []

    filename = f'./plotting/goals/{domain}-{exp_mode}-goals.pkl'
    idx_list, _, _, _ = pickle.load(open(filename, 'rb'))

    num_test_tasks = 8

    for i in range(num_test_tasks) :
        
        wd_relative_dir = f'wd_task_results/{domain}/seed_{seed}/max_path_length_{max_path_length}/interactions_{domain_to_epoch[domain] + 10}k/{exp_mode}/goal_{i}/progress.csv'
        with_ensemble_dir = os.path.join('./sac_with_initialization/data', wd_relative_dir)
        without_ensemble_dir = os.path.join('./sac_with_initialization/data_without_ensemble', wd_relative_dir)

        with_ensemble_progress = pd.read_csv(with_ensemble_dir)
        without_ensemble_progress = pd.read_csv(without_ensemble_dir)

        returns = with_ensemble_progress['remote_evaluation/Average Returns'].to_numpy()
        with_ensemble_returns.append(returns)

        returns = without_ensemble_progress['remote_evaluation/Average Returns'].to_numpy()
        without_ensemble_returns.append(returns)

    for i in idx_list:

        conven_dir = f'./oac-explore/data/train_task_results/{domain}/seed_{seed}/max_path_length_{max_path_length}/interactions_{domain_to_epoch[domain] + 10}k/{exp_mode}/goal_{i}/progress.csv'
        conven_progress = pd.read_csv(conven_dir)
        
        returns = conven_progress['remote_evaluation/Average Returns'].to_numpy()
        conven_returns.append(returns)
    
    plt.figure()
    conven_returns = np.array(conven_returns)
    conven_returns = conven_returns[:, :350].T

    with_ensemble_returns = np.array(with_ensemble_returns)
    with_ensemble_returns = with_ensemble_returns[:, :350].T

    without_ensemble_returns = np.array(without_ensemble_returns)
    without_ensemble_returns = without_ensemble_returns[:, :350].T

    plot(conven_returns, 'Randomly init SAC', color=[0, 0, 0.8, 1])
    plot(with_ensemble_returns, 'SAC init by our method', color=[1, 0, 0, 0.8])
    plot(without_ensemble_returns, 'Without ensemble', color=[0, 0.8, 0, 1])
    
    plt.xlabel('Interactions (K)', fontsize=16)
    plt.ylabel('Average returns', fontsize=12)
    plt.title('HumanoidDir', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.savefig(f'./paper_figs/ablate-ensemble-HumanoidDir.png', bbox_inches='tight')
    plt.show()

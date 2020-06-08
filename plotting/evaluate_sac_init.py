import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import importlib
import pickle
import os

from plotting.plot_utils import plot, domain_to_title, domain_to_epoch


if __name__ == "__main__":

    domain_name = ['ant-dir', 'ant-goal', 'humanoid-openai-dir', 'humanoid-goal', 'halfcheetah-vel', 'walker-param']
    for domain in domain_name:
        variant_module = importlib.import_module(f'sac_with_initialization.configs.{domain}')
        variant = variant_module.params

        seed = 0
        exp_mode = variant['exp_mode']
        max_path_length = variant['max_path_length']

        filename = f'./plotting/goals/{domain}-{exp_mode}-goals.pkl'
        idx_list, _, _, _ = pickle.load(open(filename, 'rb'))

        with_init_returns = []
        conven_returns = []
        baseline_returns = []

        num_test_tasks = 8

        for i in range(num_test_tasks) :
            # Directory to the progress.csv of each experiments
            
            wd_relative_dir = f'data/wd_task_results/{domain}/seed_{seed}/max_path_length_{max_path_length}/interactions_{domain_to_epoch[domain] + 10}k/{exp_mode}/goal_{i}/progress.csv'
            with_init_dir = os.path.join('sac_with_initialization', wd_relative_dir)
            baseline_dir = os.path.join('sac_baseline', wd_relative_dir)

            with_init_progress = pd.read_csv(with_init_dir)
            baseline_progress = pd.read_csv(baseline_dir)

            returns = with_init_progress['remote_evaluation/Average Returns'].to_numpy()
            with_init_returns.append(returns)

            returns = baseline_progress['remote_evaluation/Average Returns'].to_numpy()
            baseline_returns.append(returns)

        for i in idx_list:

            # Directory to the progress.csv of each experiments
            
            conven_dir = f'./oac-explore/data/train_task_results/{domain}/seed_{seed}/max_path_length_{max_path_length}/interactions_{domain_to_epoch[domain] + 10}k/{exp_mode}/goal_{i}/progress.csv'
            conven_progress = pd.read_csv(conven_dir)
            
            returns = conven_progress['remote_evaluation/Average Returns'].to_numpy()
            conven_returns.append(returns)
        
        plt.figure()
        conven_returns = np.array(conven_returns)
        conven_returns = conven_returns[:, :350].T

        with_init_returns = np.array(with_init_returns)
        with_init_returns = with_init_returns[:, :350].T

        plot(conven_returns, 'Randomly init SAC', color=[0, 0, 1, 1])
        plot(with_init_returns, 'SAC init by our method', color=[1, 0, 0, 1])
        
        # plt.xlabel('Interactions (K)', fontsize=16)
        # plt.ylabel('Average returns', fontsize=20)
        plt.title(domain_to_title[domain], fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'./paper_figs/init-sac-vs-train-AntDir.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        baseline_returns = np.array(baseline_returns)
        baseline_returns = baseline_returns[:, :350].T

        with_init_returns = np.array(with_init_returns)
        with_init_returns = with_init_returns[:, :350].T

        plot(baseline_returns, 'Randomly init SAC', color=[0, 0, 1, 1])
        plot(with_init_returns, 'SAC init by our method', color=[1, 0, 0, 1])
        
        if domain == 'humanoid-goal':
            plt.legend(fontsize=20)
        
        plt.xlabel('Interactions (K)', fontsize=16)
        plt.ylabel('Average returns', fontsize=20)
        plt.title(domain_to_title[domain], fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'./paper_figs/init-sac-vs-wd-AntDir.png', bbox_inches='tight')
        plt.show()

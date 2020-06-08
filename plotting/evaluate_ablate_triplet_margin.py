import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib

from plotting.plot_utils import get_results, plot, domain_to_title


if __name__ == "__main__":
    domain_name = ['ant-dir', 'ant-goal', 'humanoid-openai-dir', 'humanoid-ndone-goal', 'walker-param']
    for domain in domain_name:
        if domain != 'walker-param':
            variant_module = importlib.import_module(f'full_model.configs.{domain}')
        else:
            variant_module = importlib.import_module(f'full_model_walker_param.configs.{domain}')
        variant = variant_module.variant

        seed = variant['seed']
        exp_mode = variant['exp_mode']
        max_path_length = variant['max_path_length']
        bcq_interactions = variant['bcq_interactions']
        num_tasks = variant['algo_params']['num_tasks']

        # Directory to the progress.csv of each experiments
        triplet_margin_dirs = []
        for i in [0, 2, 4, 8]:
            relative_dir = f'data_triplet_margin_{i}p0/test_output/{domain}/mode_{exp_mode}/max_path_length_{max_path_length}/count_{num_tasks}/seed_{seed}/progress.csv'
            if domain != 'walker-param':
                full_model_dir = os.path.join('full_model', relative_dir)
            else:
                full_model_dir = os.path.join('full_model_walker_param', relative_dir)

            triplet_margin_dirs.append(full_model_dir)
        
        margin_0p0_progress = pd.read_csv(triplet_margin_dirs[0])
        margin_2p0_progress = pd.read_csv(triplet_margin_dirs[1])
        margin_4p0_progress = pd.read_csv(triplet_margin_dirs[2])
        margin_8p0_progress = pd.read_csv(triplet_margin_dirs[3])

        wd_return_smooth_0p0 = get_results(margin_0p0_progress)
        wd_return_smooth_2p0 = get_results(margin_2p0_progress)
        wd_return_smooth_4p0 = get_results(margin_4p0_progress)
        wd_return_smooth_8p0 = get_results(margin_8p0_progress)

        plt.figure(figsize=(6, 4))

        plot(wd_return_smooth_0p0[:600], 'margin = 0.0', color=[0.8, 0, 0, 1])
        plot(wd_return_smooth_2p0[:600], 'margin = 2.0', color=[0, 0, 0.8, 1])
        plot(wd_return_smooth_4p0[:600], 'margin = 4.0', color=[0, 0.8, 0, 1])
        plot(wd_return_smooth_8p0[:600], 'margin = 8.0', color=[1, 1, 0, 1])

        plt.xlabel('Epochs', fontsize=18)
        
        if 'dir' in domain:
            plt.ylabel('Average returns', fontsize=18)
            
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if domain == 'ant-goal':
            plt.legend(fontsize=16)
        plt.title(domain_to_title[domain], fontsize=24)
        plt.savefig(f'./paper_figs/wd-margin-{domain}.png', bbox_inches='tight')
        plt.show()

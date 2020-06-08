import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib

from plotting.plot_utils import get_results, get_pearl_results, plot, domain_to_title


if __name__ == "__main__":
    domain_name = ['ant-dir', 'ant-goal', 'humanoid-openai-dir', 'humanoid-ndone-goal', 'halfcheetah-vel', 'walker-param']
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
        base_log_dir = variant['base_log_dir']

        # Directory to the progress.csv of each experiments
        # We assume "base_log_dir" should be the same across methods
        relative_dir = f'{base_log_dir}/test_output/{domain}/mode_{exp_mode}/max_path_length_{max_path_length}/count_{num_tasks}/seed_{seed}/progress.csv'

        contextual_bcq_dir = os.path.join('contextual_bcq', relative_dir)
        batch_pearl_dir = f'./batch_pearl/output/{domain}/progress.csv'
        if domain != 'walker-param':
            full_model_dir = os.path.join('full_model', relative_dir)
        else:
            full_model_dir = os.path.join('full_model_walker_param', relative_dir)

        full_model_progress = pd.read_csv(full_model_dir)
        contextual_bcq_progress = pd.read_csv(contextual_bcq_dir)
        batch_pearl_progress = pd.read_csv(batch_pearl_dir)

        wd_return_smooth_full_model =  get_results(full_model_progress)
        wd_return_smooth_bcq_encoder = get_results(contextual_bcq_progress, bcq_encoder=True)
        wd_return_smooth_pearl = get_pearl_results(batch_pearl_progress)

        plt.figure(figsize=(6, 4))
        plot(wd_return_smooth_bcq_encoder[:600], 'Contextual BCQ', color=[0, 0, 0.8, 1])

        plot(wd_return_smooth_full_model[:600], 'Our model', color=[0.8, 0, 0, 1])

        plot(wd_return_smooth_pearl[:600], 'PEARL', color=[0, 0.8, 0, 1])
            
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if domain_to_title[domain] == 'HumanoidGoal':
            plt.legend(fontsize=20)
            
        plt.title(domain_to_title[domain], fontsize=24)
        plt.savefig(f'./paper_figs/wd-{domain}.png', bbox_inches='tight')
        plt.show()


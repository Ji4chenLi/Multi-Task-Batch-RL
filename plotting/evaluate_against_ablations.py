import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib


from plotting.plot_utils import get_results, plot, domain_to_title


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

        no_trans_relabel_dir = os.path.join('no_transition_relabelling', relative_dir)
        neither_dir = os.path.join('neither', relative_dir)
        if domain != 'walker-param':
            full_model_dir = os.path.join('full_model', relative_dir)
            no_triplet_loss_dir = os.path.join('no_triplet_loss', relative_dir)
        else:
            full_model_dir = os.path.join('full_model_walker_param', relative_dir)
            no_triplet_loss_dir = os.path.join('no_triplet_loss_walker_param', relative_dir)

        full_model_progress = pd.read_csv(full_model_dir)
        no_trans_relabel_progress = pd.read_csv(no_trans_relabel_dir)
        no_triplet_loss_progress = pd.read_csv(no_triplet_loss_dir)
        neither_progress = pd.read_csv(neither_dir)

        wd_return_smooth_full_model = get_results(full_model_progress)
        wd_return_smooth_no_trans_relabel = get_results(no_trans_relabel_progress)
        wd_return_smooth_no_triplet_loss = get_results(no_triplet_loss_progress)
        wd_return_smooth_neither = get_results(neither_progress)

        plt.figure(figsize=(6, 4))
        plot(wd_return_smooth_full_model[:600], 'Full model', color=[0.8, 0, 0, 1])
    
        plot(wd_return_smooth_no_trans_relabel[:600], 'No transition relabelling', color=[0, 0, 0.8, 1])

        plot(wd_return_smooth_no_triplet_loss[:600], 'No triplet loss', color=[0, 0.8, 0, 1])
        
        plot(wd_return_smooth_neither[:600], 'Neither', color=[1, 1, 0, 1])

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if domain_to_title[domain] == 'HumanoidGoal':
            plt.legend(fontsize=16)
        plt.title(domain_to_title[domain], fontsize=24)
        plt.savefig(f'./paper_figs/wd-ablation-{domain}.png', bbox_inches='tight')
        plt.show()

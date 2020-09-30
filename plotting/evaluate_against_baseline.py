import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib
import pylab

from plotting.plot_utils import get_results, get_pearl_results, plot, domain_to_title


if __name__ == "__main__":
    domain_name = ['ant-dir', 'ant-goal', 'humanoid-openai-dir', 'halfcheetah-vel', 'walker-param', 'maze-umaze']
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
        plot(wd_return_smooth_full_model[:600], 'Our model', color=[0.8, 0, 0, 1])

        plot(wd_return_smooth_bcq_encoder[:600], 'Contextual BCQ', color=[0, 0, 0.8, 1])

        plot(wd_return_smooth_pearl[:600], 'PEARL', color=[0, 0.8, 0, 1])

        max_full_model = np.mean(wd_return_smooth_full_model[:600], axis=1).max()
        max_bcq_encoder = np.mean(wd_return_smooth_bcq_encoder[:600], axis=1).max()
        max_pearl = np.mean(wd_return_smooth_pearl[:600], axis=1).max()
        
        gain_bcq_encoder = np.round(abs((max_full_model - max_bcq_encoder) / max_bcq_encoder), 2)
        gain_pearl = np.round(abs((max_full_model - max_pearl) / max_pearl), 2)
            
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.title(f'{domain_to_title[domain]}\n MBML (Ours) +{int(gain_bcq_encoder * 100)}%, +{int(gain_pearl * 100)}%\nover Contextual BCQ, PEARL', fontsize=24)    
        plt.savefig(f'./paper_figs/wd-{domain_to_title[domain]}.png', bbox_inches='tight')
        plt.show()

    # Generate legends
    
    x = np.arange(10)

    # create a figure for the data
    figData = pylab.figure()
    ax = pylab.gca()

    pylab.plot(x, x * (0+1), label='MBML (Ours)', color=[0.8, 0, 0, 1], linewidth=8.0)

    pylab.plot(x, x * (1+1), label='Contextual BCQ', color=[0, 0, 0.8, 1], linewidth=8.0)

    pylab.plot(x, x * (2+1), label='PEARL', color=[0, 0.8, 0, 1], linewidth=8.0)

    # create a second figure for the legend
    figLegend = pylab.figure(figsize = (6, 3))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', fontsize=50, labelspacing=1.0)

    # save the legend to files
    figLegend.savefig("./paper_figs/legend_ablation.png")



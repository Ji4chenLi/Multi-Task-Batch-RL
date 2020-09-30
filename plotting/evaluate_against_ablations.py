import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib
import pylab

from plotting.plot_utils import get_results, plot, domain_to_title


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

        max_full_model = np.mean(wd_return_smooth_full_model[:600], axis=1).max()
        max_no_relabel = np.mean(wd_return_smooth_no_trans_relabel[:600], axis=1).max()
        max_no_triplet = np.mean(wd_return_smooth_no_triplet_loss[:600], axis=1).max()
        max_neither = np.mean(wd_return_smooth_neither[:600], axis=1).max()
        
        gain_no_relabel = np.round(abs((max_full_model - max_no_relabel) / max_no_relabel), 2)
        gain_no_triplet = np.round(abs((max_full_model - max_no_triplet) / max_no_triplet), 2)
        gain_neither = np.round(abs((max_full_model - max_neither) / max_neither), 2)
        
        if domain == 'maze-umaze':
            gt_dir = os.path.join('full_model_ground_truth_label', relative_dir)
            gt_progress = pd.read_csv(gt_dir)
            wd_return_smooth_gt = get_results(gt_progress)

            plot(wd_return_smooth_gt[:600], 'Neither', color=[1, 1, 0, 1])
            max_gt = np.mean(wd_return_smooth_gt[:600], axis=1).max()
            gain_gt = np.round((max_full_model - max_gt) / abs(max_gt), 2)

            plt.title(f'{domain_to_title[domain]}\nMBML +{int(gain_no_relabel * 100)}%, +{int(gain_no_triplet * 100)}%, +{int(gain_neither * 100)}, {int(gain_gt * 100)}%\nover NR, NT, Neither, GT', fontsize=24)

        plt.title(f'{domain_to_title[domain]}\nMBML (Ours) +{int(gain_no_relabel * 100)}%, +{int(gain_no_triplet * 100)}%, +{int(gain_neither * 100)}%\nover NR, NT, Neither', fontsize=24)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'./paper_figs/wd-ablation-{domain_to_title[domain]}.png', bbox_inches='tight')
        plt.show()
    
    # Generate legends

    
    x = np.arange(10)

    # create a figure for the data
    figData = pylab.figure()
    ax = pylab.gca()

    pylab.plot(x, x * (0+1), label='MBML (Ours)', color=[0.8, 0, 0, 1], linewidth=8.0)

    pylab.plot(x, x * (1+1), label='No relabelling (NR)', color=[0, 0, 0.8, 1], linewidth=8.0)

    pylab.plot(x, x * (2+1), label='No triplet loss (NT)', color=[0, 0.8, 0, 1], linewidth=8.0)

    pylab.plot(x, x * (3+1), label='Neither', color=[1, 1, 0, 1], linewidth=8.0)

    pylab.plot(x, x * (4+1), label='Ground truth\nreward relabel (GT)', color=[0.8, 0.5, 0, 1], linewidth=8.0)

    # create a second figure for the legend
    figLegend = pylab.figure(figsize = (10, 7.55))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', fontsize=50, labelspacing=1.0)

    # save the legend to files
    figLegend.savefig("./paper_figs/legend_ablation.png")



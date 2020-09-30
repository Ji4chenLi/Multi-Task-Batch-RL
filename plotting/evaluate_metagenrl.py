import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('./metagenrl/custom_mean_episode_reward.csv')

    steps = df[['Step']].values.flatten()
    values = df[['Value']].values.flatten()

    # Multiply by 10 because we perform 10 grad step every 600 timesteps 
    # Divide by 100 because in the experiment in our paper, 1 epoch equals 100 grad steps
    epochs = (steps / 600) * 10 / 100
    mask = epochs < 200
    epochs = epochs[mask]
    plt.figure(figsize=(6, 4))

    plt.plot(epochs, values[:sum(mask)], label='MetaGenRL')

    plt.hlines(xmin=0, xmax=200, y=767, label='Our model\nat 200 epochs', colors='red')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Avg. returns', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=22, loc='right')

    plt.title('AntDir training task', fontsize=24)
    # plt.tight_layout()
    plt.savefig('./paper_figs/metagenrl.png', bbox_inches='tight')
    plt.show()
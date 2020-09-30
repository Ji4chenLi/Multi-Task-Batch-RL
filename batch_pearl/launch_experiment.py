"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import pickle
import socket

from replay_buffer import ReplayBuffer, MultiTaskReplayBuffer
from utils.env_utils import env_producer, domain_to_epoch, domain_to_num_goals
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
from rlkit.launchers.launcher_util import set_seed
import rlkit.torch.pytorch_util as ptu
from utils.rng import set_seed
from configs.default import default_config
import logging 
import ray
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.WARNING,

    # The amount of memory (in bytes)
    # object_store_memory=1073741824 * 2, # 2g
    redis_max_memory=1073741824 * 20 # 2g
)


def experiment(variant):

    domain = variant['domain']
    seed = variant['seed']
    exp_mode = variant['exp_mode']
    max_path_length = variant['algo_params']['max_path_length']
    bcq_interactions = variant['bcq_interactions']
    num_tasks = variant['num_tasks']

    filename = f'./goals/{domain}-{exp_mode}-goals.pkl'
    idx_list, train_goals, wd_goals, ood_goals = pickle.load(open(filename, 'rb'))
    idx_list = idx_list[:num_tasks]

    sub_buffer_dir = f"buffers/{domain}/{exp_mode}/max_path_length_{max_path_length}/interactions_{bcq_interactions}k/seed_{seed}"
    buffer_dir = os.path.join(variant['data_models_root'], sub_buffer_dir)

    print("Buffer directory: " + buffer_dir)


    # Load buffer
    bcq_buffers = []

    buffer_loader_id_list = []
    for i, idx in enumerate(idx_list):
        bname = f'goal_0{idx}.zip_pkl' if idx < 10 else f'goal_{idx}.zip_pkl'
        filename = os.path.join(buffer_dir, bname)
        rp_buffer = ReplayBuffer.remote(
            index=i,
            seed=seed,
            num_trans_context=variant['num_trans_context'],
            in_mdp_batch_size=variant['in_mdp_batch_size'],
        )
        
        buffer_loader_id_list.append(rp_buffer.load_from_gzip.remote(filename))
        bcq_buffers.append(rp_buffer)
    ray.get(buffer_loader_id_list)

    assert len(bcq_buffers) == len(idx_list)

    train_buffer = MultiTaskReplayBuffer(
        bcq_buffers_list=bcq_buffers,
    )

    set_seed(variant['seed'])

    # create multi-task environment and sample tasks
    env = env_producer(variant['domain'], seed=0)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_goals=train_goals,
        wd_goals=wd_goals,
        ood_goals=ood_goals,
        replay_buffers=train_buffer,
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['domain'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default='./configs/ant-dir.json')
@click.option('--data_models_root', default='../data_and_trained_models')
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)


def main(config, data_models_root, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    variant['data_models_root'] = data_models_root

    experiment(variant)


if __name__ == "__main__":
    main()


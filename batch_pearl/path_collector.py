import numpy as np
import torch
import ray
from collections import deque
from BCQ_plus_encoder import BCQ

from utils.env_utils import env_producer
import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy, get_numpy

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder


@ray.remote(num_cpus=1.0, num_gpus=0.2)
class RemotePathCollectorSingleMdp(object):
    def __init__(self, variant, goal, candidate_size=10):
        ptu.set_gpu_mode(True)
        torch.set_num_threads(1)

        import sys
        sys.argv = ['']
        del sys

        self.env = env_producer(variant['env_name'], seed=0, goal=goal)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
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

        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
        )
        self.num_evals = variant['num_evals']
        self.max_path_length = variant['max_path_length']

    def async_evaluate(self, params_list, goal=None):
        if goal is not None:
            self.env.set_goal(goal)
        
        self.set_policy_params(params_list)
        avg_reward = 0.
        for _ in range(self.num_evals):
            obs = self.env.reset()
            done = False
            path_length = 0
            raw_context = deque(maxlen=self.max_path_length)
            while not done and path_length < self.max_path_length:
                action = self.select_actions(np.array(obs), raw_context)
                next_obs, reward, done, _ = self.env.step(action)
                raw_context.append(
                    np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1
                    )
                )
                obs = next_obs.copy()
                avg_reward += reward
                path_length += 1

        avg_reward /= self.num_evals
        return avg_reward
    
    def set_policy_params(self, params_list):
        '''
        The shipped params are in cpu here. This function
        will set the params of the sampler's networks using
        the params in the params_list and ship them to gpu.
        '''
        encoder_params, policy_params = params_list
        
        self.agent.context_encoder.load_state_dict(encoder_params)
        self.agent.context_encoder.to(ptu.device)

        self.agent.policy.load_state_dict(policy_params)
        self.agent.policy.to(ptu.device)
        
    def select_actions(self, obs, raw_context):

        # Repeat the obs as what BCQ has done, 
        # candidate_size here indicates how many
        # candidate actions we need.
        if len(raw_context) == 0:
            # In the beginning, the inferred_mdp is set to zero vector.
            inferred_mdp = ptu.zeros((1, self.policy.mlp_encoder.encoder_latent_dim))
        else:
            # Construct the context from raw context
            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            inferred_mdp = self.policy.mlp_encoder(context)
            
        # obs = torch.cat([obs, inferred_mdp], dim=1)
        action = self.policy.select_action(obs, get_numpy(inferred_mdp))

        return action


@ray.remote(num_cpus=1.0)
class RemotePathCollector(object):
    def __init__(self, variant, eval_goals_set):
        ptu.set_gpu_mode(True)
        torch.set_num_threads(1)

        self._single_mdp_path_collectors = [
            RemotePathCollectorSingleMdp.remote(
                variant, goal
            ) for goal in eval_goals_set
        ]

    def async_evaluate(self, params_list, goals=None):
        if goals is not None:
            return_obj_ids = [
                spc.async_evaluate.remote(params_list, goals[i]) for i, spc in enumerate(self._single_mdp_path_collectors)
            ]
        else:
            return_obj_ids = [
                spc.async_evaluate.remote(params_list) for spc in self._single_mdp_path_collectors
            ]
        return ray.get(return_obj_ids)

    def end_epoch(self, epoch):
        pass
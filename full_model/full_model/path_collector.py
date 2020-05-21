import numpy as np
import torch
import ray
from collections import deque

from utils.env_utils import env_producer
import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy
from utils.rng import set_seed

from networks import FlattenMlp, MlpEncoder, VaeDecoder, PerturbationGenerator
from prob_context_encoder import ProbabilisticContextEncoder


@ray.remote(num_gpus=0.4)
class RemotePathCollectorSingleMdp(object):
    def __init__(self, index, variant, candidate_size=10):
        ptu.set_gpu_mode(True)
        torch.set_num_threads(1)

        import sys
        sys.argv = ['']
        del sys

        env_max_action = variant['env_max_action']
        obs_dim = variant['obs_dim']
        action_dim = variant['action_dim']
        latent_dim = variant['latent_dim']
        vae_latent_dim = 2 * action_dim
        mlp_enconder_input_size = 2 * obs_dim + action_dim + 1 if variant['use_next_obs_in_context'] else obs_dim + action_dim + 1

        mlp_enconder = MlpEncoder(
            hidden_sizes=[200, 200, 200],
            input_size=mlp_enconder_input_size,
            output_size=2 * variant['latent_dim']

        )
        self.context_encoder = ProbabilisticContextEncoder(
            mlp_enconder,
            variant['latent_dim']
        )
        self.Qs = FlattenMlp(
            hidden_sizes=variant['Qs_hidden_sizes'],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
        )
        self.vae_decoder = VaeDecoder(
            max_action=variant['env_max_action'],
            hidden_sizes=variant['vae_hidden_sizes'],
            input_size=obs_dim + vae_latent_dim + latent_dim,
            output_size=action_dim,
        )
        self.perturbation_generator = PerturbationGenerator(
            max_action=env_max_action,
            hidden_sizes=variant['perturbation_hidden_sizes'],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=action_dim,
        )

        self.use_next_obs_in_context = variant['use_next_obs_in_context']

        self.env = env_producer(variant['domain'], variant['seed'])
        self.num_evals = variant['num_evals']
        self.max_path_length = variant['max_path_length']

        self.vae_latent_dim = vae_latent_dim
        self.candidate_size = variant['candidate_size']
        
        self.env.seed(10 * variant['seed'] + 1234 + index)
        set_seed(10 * variant['seed'] + 1234 + index)

        self.env.action_space.np_random.seed(123 + index)

    def async_evaluate(self, goal):
        self.env.set_goal(goal)
        self.context_encoder.clear_z()

        avg_reward = 0.
        avg_achieved = []
        final_achieved = []

        raw_context = deque()
        for i in range(self.num_evals):
            # Sample MDP indentity
            self.context_encoder.sample_z()
            inferred_mdp = self.context_encoder.z

            obs = self.env.reset()
            done = False
            path_length = 0
            
            while not done and path_length < self.max_path_length:
                action = self.select_actions(np.array(obs), inferred_mdp)
                next_obs, reward, done, env_info = self.env.step(action)
                avg_achieved.append(env_info['achieved'])
                if self.use_next_obs_in_context:
                    new_context = np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), next_obs.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1)
                else:
                    assert False
                    new_context = np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1)
                raw_context.append(new_context)
                obs = next_obs.copy()
                if i > 1:
                    avg_reward += reward
                path_length += 1

            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            self.context_encoder.infer_posterior(context)
            
            if i > 1:
                final_achieved.append(env_info['achieved'])

        avg_reward /= (self.num_evals - 2)
        if np.isscalar(env_info['achieved']):
            avg_achieved = np.mean(avg_achieved)
            final_achieved = np.mean(final_achieved)

        else:
            avg_achieved = np.stack(avg_achieved)
            avg_achieved = np.mean(avg_achieved, axis=0)

            final_achieved = np.stack(final_achieved)
            final_achieved = np.mean(final_achieved, axis=0)
        print(avg_reward)
        return avg_reward, (final_achieved.tolist(), self.env._goal.tolist())
    
    def set_network_params(self, params_list):
        '''
        The shipped params are in cpu here. This function
        will set the params of the sampler's networks using
        the params in the params_list and ship them to gpu.
        '''
        context_encoder_params, Qs_params, vae_params, perturbation_params = params_list
        
        self.context_encoder.mlp_encoder.set_param_values(context_encoder_params)
        self.context_encoder.mlp_encoder.to(ptu.device)

        self.Qs.set_param_values(Qs_params)
        self.Qs.to(ptu.device)

        self.vae_decoder.set_param_values(vae_params)
        self.vae_decoder.to(ptu.device)

        self.perturbation_generator.set_param_values(perturbation_params)
        self.perturbation_generator.to(ptu.device)

    def select_actions(self, obs, inferred_mdp):

        # Repeat the obs as what BCQ has done, 
        # candidate_size here indicates how many
        # candidate actions we need.
        obs = from_numpy(np.tile(
            obs.reshape(1, -1), (self.candidate_size, 1)
        ))
        with torch.no_grad():
            inferred_mdp = inferred_mdp.repeat(self.candidate_size, 1)
            z = from_numpy(
                np.random.normal(0, 1, size=(obs.size(0), self.vae_latent_dim))
            ).clamp(-0.5, 0.5).to(ptu.device)
            candidate_actions = self.vae_decoder(
                obs, z, inferred_mdp
            )
            perturbed_actions = self.perturbation_generator.get_perturbed_actions(
                obs, candidate_actions, inferred_mdp
            )
            qv = self.Qs(obs, perturbed_actions, inferred_mdp)
            ind = qv.max(0)[1]
        return ptu.get_numpy(perturbed_actions[ind])


class RemotePathCollector(object):
    def __init__(self, variant):
        num_workers = variant['algo_params']['num_workers']
        self._single_mdp_path_collectors = [
            RemotePathCollectorSingleMdp.remote(
                i, variant
            ) for i in range(num_workers)
        ]

    def async_evaluate(self, goals=None):
        return_obj_ids = [
            self._single_mdp_path_collectors[i].async_evaluate.remote(goal) for i, goal in enumerate(goals)
        ]

        return return_obj_ids

    def async_evaluate_test(self, goals=None):
        return_obj_ids = [
            self._single_mdp_path_collectors[i].async_evaluate_test.remote(goal) for i, goal in enumerate(goals)
        ]

        return return_obj_ids

    def set_network_params(self, params_list):

        ids = []
        for spc in self._single_mdp_path_collectors:
            ray_id = spc.set_network_params.remote(params_list)
            ids.append(ray_id)

        ray.get(ids)

    def end_epoch(self, epoch):
        pass



# def async_evaluate_test(self, goal):
#         self.env.set_goal(goal)
#         self.context_encoder.clear_z()

#         avg_reward_list = []
#         final_achieved = []

#         raw_context = deque()
#         for _ in range(self.num_evals):
#             # Sample MDP indentity
#             self.context_encoder.sample_z()
#             inferred_mdp = self.context_encoder.z

#             obs = self.env.reset()
#             done = False
#             path_length = 0
#             avg_reward = 0
#             while not done and path_length < self.max_path_length:
#                 action = self.select_actions(np.array(obs), inferred_mdp)
#                 next_obs, reward, done, env_info = self.env.step(action)
#                 if self.use_next_obs_in_context:
#                     new_context = np.concatenate([
#                         obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1), next_obs.reshape(1, -1)
#                     ], axis=1)
                    
#                 else:
#                     new_context = np.concatenate([
#                         obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
#                     ], axis=1)
#                 raw_context.append(new_context)
#                 obs = next_obs.copy()
#                 avg_reward += reward
#                 path_length += 1

#             avg_reward_list.append(avg_reward)
#             final_achieved.append(env_info['achieved'])

#             context = from_numpy(np.concatenate(raw_context, axis=0))[None]
#             self.context_encoder.infer_posterior(context)

#         return avg_reward_list, final_achieved
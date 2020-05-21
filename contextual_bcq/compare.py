import numpy as np
import torch
import ray
from collections import deque

from utils.env_utils import env_producer
import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy
from utils.rng import set_seed

from networks import FlattenMlp, MlpEncoder, VaeDecoder, PerturbationGenerator


@ray.remote(num_cpus=1.0, num_gpus=0.4)
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

        self.f = MlpEncoder(
            g_hidden_sizes=variant['g_hidden_sizes'],
            g_input_sizes=obs_dim + action_dim + 1,
            g_latent_dim=variant['g_latent_dim'],
            h_hidden_sizes=variant['h_hidden_sizes'],
            latent_dim=latent_dim,
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

        self.env = env_producer(variant['domain'], variant['seed'])
        self.num_evals = variant['algo_params']['num_evals']
        self.max_path_length = variant['max_path_length']

        self.vae_latent_dim = vae_latent_dim
        self.num_trans_context = variant['num_trans_context']
        self.candidate_size = variant['candidate_size']
        self.seed = variant['seed']
        self.index = index
        
        self.env.seed(10 * self.seed + 1234 + index)
        set_seed(10 * self.seed + 1234 + index)

    def async_evaluate(self, goal):
        self.env.set_goal(goal)

        avg_reward = 0.
        avg_achieved = []
        final_achieved = []
        
        for _ in range(self.num_evals):
            obs = self.env.reset()
            done = False
            path_length = 0
            raw_context = deque()
            while not done and path_length < self.max_path_length:
                action = self.select_actions(np.array(obs), raw_context)
                next_obs, reward, done, env_info = self.env.step(action)
                avg_achieved.append(env_info['achieved'])
                raw_context.append(
                    np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1
                    )
                )
                obs = next_obs.copy()
                avg_reward += reward
                path_length += 1
            final_achieved.append(env_info['achieved'])

        avg_reward /= self.num_evals
        if np.isscalar(env_info['achieved']):
            avg_achieved = np.mean(avg_achieved)
            final_achieved = np.mean(final_achieved)

        else:
            avg_achieved = np.stack(avg_achieved)
            avg_achieved = np.mean(avg_achieved, axis=0)

            final_achieved = np.stack(final_achieved)
            final_achieved = np.mean(final_achieved, axis=0)
        print(avg_reward)
        print(type(final_achieved), type(self.env._goal))
        return avg_reward, (final_achieved.tolist(), self.env._goal.tolist())

    def get_rollout(self, goal=None):
        self.env.set_goal(goal)

        obs = self.env.reset()
        done = False
        path_length = 0
        traj = []
        raw_context = deque()
        while not done and path_length < self.max_path_length:
            action, inferred_mdp = self.select_actions(np.array(obs), raw_context)
            next_obs, reward, done, env_info = self.env.step(action)
            traj.append([obs, next_obs, action, reward, inferred_mdp, env_info])
            raw_context.append(
                np.concatenate([
                    obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                ], axis=1
                )
            )
            obs = next_obs.copy()
            path_length += 1
        return traj
    
    def set_network_params(self, params_list):
        '''
        The shipped params are in cpu here. This function
        will set the params of the sampler's networks using
        the params in the params_list and ship them to gpu.
        '''
        f_params, Qs_params, vae_params, perturbation_params = params_list
        
        self.f.set_param_values(f_params)
        self.f.to(ptu.device)

        self.Qs.set_param_values(Qs_params)
        self.Qs.to(ptu.device)

        self.vae_decoder.set_param_values(vae_params)
        self.vae_decoder.to(ptu.device)

        self.perturbation_generator.set_param_values(perturbation_params)
        self.perturbation_generator.to(ptu.device)

    def select_actions(self, obs, raw_context):

        # Repeat the obs as what BCQ has done, 
        # candidate_size here indicates how many
        # candidate actions we need.
        obs = from_numpy(np.tile(
            obs.reshape(1, -1), (self.candidate_size, 1)
        ))
        if len(raw_context) == 0:
            # In the beginning, the inferred_mdp is set to zero vector.
            inferred_mdp = ptu.zeros((1, self.f.latent_dim))
        else:
            # Construct the context from raw context
            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            inferred_mdp = self.f(context)
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
        if goals is not None:
            return_obj_ids = [
                self._single_mdp_path_collectors[i].async_evaluate.remote(goal) for i, goal in enumerate(goals)
            ]
        else:
            return_obj_ids = [
                spc.async_evaluate.remote() for spc in self._single_mdp_path_collectors
            ]
        return return_obj_ids

    def set_network_params(self, params_list):
        for spc in self._single_mdp_path_collectors:
            ray.get(spc.set_network_params.remote(params_list))

    def get_rollout(self, goals):
        return_list = [
            spc.get_rollout.remote(goals[i]) for i, spc in enumerate(self._single_mdp_path_collectors)
        ]
        return return_list

    def end_epoch(self, epoch):
        pass


class PathCollector(object):
    def __init__(self, variant, eval_goals_set):
        ptu.set_gpu_mode(True)
        
        self._single_mdp_path_collectors = [
            PathCollectorSingleMdp(
                variant, goal
            ) for goal in eval_goals_set
        ]

    def async_evaluate(self, params_list, goals=None):
        if goals is not None:
            return_list = [
                spc.async_evaluate(params_list, goals[i]) for i, spc in enumerate(self._single_mdp_path_collectors)
            ]
        else:
            return_list = [
                spc.async_evaluate(params_list) for spc in self._single_mdp_path_collectors
            ]
        return return_list


    def end_epoch(self, epoch):
        pass


class PathCollectorSingleMdp(object):
    def __init__(self, variant, goal, candidate_size=10):
        ptu.set_gpu_mode(True)

        import sys
        sys.argv = ['']
        del sys

        env_max_action = variant['env_max_action']
        obs_dim = variant['obs_dim']
        action_dim = variant['action_dim']
        latent_dim = variant['latent_dim']
        vae_latent_dim = 2 * action_dim

        self.f = MlpEncoder(
            g_hidden_sizes=variant['g_hidden_sizes'],
            g_input_sizes=obs_dim + action_dim + 1,
            g_latent_dim=variant['g_latent_dim'],
            h_hidden_sizes=variant['h_hidden_sizes'],
            latent_dim=latent_dim,
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

        self.env = env_producer(variant['domain'], variant['seed'], goal)
        self.num_evals = variant['algo_params']['num_evals']
        self.max_path_length = variant['algo_params']['max_path_length']

        self.vae_latent_dim = vae_latent_dim
        self.num_trans_context = variant['num_trans_context']
        self.candidate_size = variant['candidate_size']

    def async_evaluate(self, params_list, goal=None):
        if goal is not None:
            self.env.set_goal(goal)
        
        self.set_network_params(params_list)
        avg_reward = 0.
        avg_achieved = []
        final_achieved = []
        for _ in range(self.num_evals):
            obs = self.env.reset()
            done = False
            path_length = 0
            raw_context = deque()
            while not done and path_length < self.max_path_length:
                action = self.select_actions(np.array(obs), raw_context)
                next_obs, reward, done, env_info = self.env.step(action)
                avg_achieved.append(env_info['achieved'])
                raw_context.append(
                    np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1
                    )
                )
                print(env_info['achieved'])
                obs = next_obs.copy()
                avg_reward += reward
                path_length += 1
            final_achieved.append(env_info['achieved'])

        avg_reward /= self.num_evals
        if np.isscalar(env_info['achieved']):
            avg_achieved = np.mean(avg_achieved)
            final_achieved = np.mean(final_achieved)
        else:
            # avg_achieved = np.stack(avg_achieved)
            # avg_achieved = np.mean(avg_achieved, axis=0)

            final_achieved = np.stack(final_achieved)
            final_achieved = np.mean(final_achieved, axis=0)

        return avg_reward, (final_achieved.tolist(), self.env._goal.tolist())
        # return avg_reward, (avg_achieved, self.env._goal), (final_achieved, self.env._goal)

    def get_rollout(self, goal=None, bcq_policy=None):
        if goal is not None:
            self.env.set_goal(goal)

        obs = self.env.reset()
        done = False
        path_length = 0
        avg_reward = 0.
        traj = []
        raw_context = deque()
        while not done and path_length < self.max_path_length:
            if bcq_policy is not None and path_length < 20:
                # print(obs[:2])
                action = bcq_policy.select_action(obs)
            else:
                # print(obs[:2])
                action = self.select_actions(np.array(obs), raw_context)
            action = self.select_actions(np.array(obs), raw_context)
            next_obs, reward, done, env_info = self.env.step(action)
            traj.append([obs, next_obs, action, reward, raw_context, env_info])
            raw_context.append(
                np.concatenate([
                    obs.reshape(1, -1), action.reshape(1, -1), np.array(reward).reshape(1, -1)
                ], axis=1
                )
            )
            obs = next_obs.copy()
            path_length += 1
            avg_reward += reward
            
        print(avg_reward)
        return traj
    
    def set_network_params(self, params_list):
        '''
        The shipped params are in cpu here. This function
        will set the params of the sampler's networks using
        the params in the params_list and ship them to gpu.
        '''
        f_params, Qs_params, vae_params, perturbation_params = params_list
        
        self.f.set_param_values(f_params)
        self.f.to(ptu.device)

        self.Qs.set_param_values(Qs_params)
        self.Qs.to(ptu.device)

        self.vae_decoder.set_param_values(vae_params)
        self.vae_decoder.to(ptu.device)

        self.perturbation_generator.set_param_values(perturbation_params)
        self.perturbation_generator.to(ptu.device)

    def select_actions(self, obs, raw_context):

        # Repeat the obs as what BCQ has done, 
        # candidate_size here indicates how many
        # candidate actions we need.
        obs = from_numpy(np.tile(
            obs.reshape(1, -1), (self.candidate_size, 1)
        ))
        if len(raw_context) == 0:
            # In the beginning, the inferred_mdp is set to zero vector.
            inferred_mdp = ptu.zeros((1, self.f.latent_dim))
        else:
            # Construct the context from raw context
            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            inferred_mdp = self.f(context)
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
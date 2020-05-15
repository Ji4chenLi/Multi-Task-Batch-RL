import numpy as np
import torch
import ray
from collections import deque
from BCQ_plus_encoder import BCQ

from utils.env_utils import env_producer
import utils.pytorch_util as ptu
from utils.pytorch_util import from_numpy, get_numpy
from utils.rng import set_seed


@ray.remote(num_gpus=0.4)
class RemotePathCollectorSingleMdp(object):
    def __init__(self, index, variant, candidate_size=10):
        ptu.set_gpu_mode(True)
        torch.set_num_threads(1)

        import sys
        sys.argv = ['']
        del sys

        self.env = env_producer(variant['domain'], variant['seed'])
        state_dim = self.env.observation_space.low.size
        action_dim = self.env.action_space.low.size
        max_action = float(self.env.action_space.high[0])

        self.policy = BCQ(state_dim, action_dim, max_action, **variant['policy_params'])
        self.num_evals = variant['num_evals']
        self.max_path_length = variant['max_path_length']
        self.seed = variant['seed']
        self.index = index
        
        self.env.seed(10 * self.seed + 1234 + index)
        set_seed(10 * self.seed + 1234 + index)

    def async_evaluate(self, goal):
        self.env.set_goal(goal)

        for param, param_val in goal.items():
            assert np.array_equal(param_val, getattr(self.env.model, param))
            
        self.policy.context_encoder.clear_z()

        avg_reward = 0.
        avg_achieved = []
        final_achieved = []

        raw_context = deque()
        for i in range(self.num_evals):
            # Sample MDP indentity
            self.policy.context_encoder.sample_z()
            inferred_mdp = self.policy.context_encoder.z

            obs = self.env.reset()
            done = False
            path_length = 0
            
            while not done and path_length < self.max_path_length:
                action = self.select_actions(np.array(obs), inferred_mdp)
                next_obs, reward, done, env_info = self.env.step(action)
                avg_achieved.append(env_info['achieved'])

                new_context = np.concatenate([
                    obs.reshape(1, -1), action.reshape(1, -1), next_obs.reshape(1, -1), np.array(reward).reshape(1, -1)
                ], axis=1)

                raw_context.append(new_context)
                obs = next_obs.copy()
                if i > 1:
                    avg_reward += reward
                path_length += 1

            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            self.policy.context_encoder.infer_posterior(context)
            
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
    
    def set_policy_params(self, params_list):
        '''
        The shipped params are in cpu here. This function
        will set the params of the sampler's networks using
        the params in the params_list and ship them to gpu.
        '''
        actor_params, critic_params, vae_params, context_encoder_params = params_list
        
        self.policy.actor.load_state_dict(actor_params)
        self.policy.actor.to(ptu.device)

        self.policy.critic.load_state_dict(critic_params)
        self.policy.actor.to(ptu.device)

        self.policy.vae.load_state_dict(vae_params)
        self.policy.vae.to(ptu.device)

        self.policy.context_encoder.mlp_encoder.load_state_dict(context_encoder_params)
        self.policy.context_encoder.mlp_encoder.to(ptu.device)

    def select_actions(self, obs, inferred_mdp):
        action = self.policy.select_action(obs, get_numpy(inferred_mdp))

        return action


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

    def set_policy_params(self, params_list):

        ids = []
        for spc in self._single_mdp_path_collectors:
            ray_id = spc.set_policy_params.remote(params_list)
            ids.append(ray_id)
        
        ray.get(ids)

    def end_epoch(self, epoch):
        pass
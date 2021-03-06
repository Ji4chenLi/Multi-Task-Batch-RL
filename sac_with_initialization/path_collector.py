from collections import deque, OrderedDict
import torch

from utils.env_utils import env_producer
from utils.eval_util import create_stats_ordered_dict
from utils.pytorch_util import from_numpy
import utils.pytorch_util as ptu
from utils.rng import get_global_pkg_rng_state, set_global_pkg_rng_state
import numpy as np
import ray
from optimistic_exploration import get_optimistic_exploration_action


class MdpPathCollector(object):
    def __init__(
            self,
            env,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):

        # The class state which we do not expect to mutate
        if render_kwargs is None:
            render_kwargs = {}
        self._render = render
        self._render_kwargs = render_kwargs
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved

        # The class mutable internal state
        self._env = env
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            policy,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            optimistic_exploration=False,
            optimistic_exploration_kwargs={}
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                policy,
                max_path_length=max_path_length_this_loop,
                optimistic_exploration=optimistic_exploration,
                optimistic_exploration_kwargs=optimistic_exploration_kwargs
            )
            path_len = len(path['actions'])
            if (
                    # incomplete path
                    path_len != max_path_length and

                    # that did not end in a terminal state
                    not path['terminals'][-1] and

                    # and we should discard such path
                    discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            # env_mj_state=self._env.sim.get_state(),
            env_rng=self._env.np_random.get_state(),

            _epoch_paths=self._epoch_paths,
            _num_steps_total=self._num_steps_total,
            _num_paths_total=self._num_paths_total
        )

    def restore_from_snapshot(self, ss):

        self._env.sim.set_state(ss['env_mj_state'])
        self._env.np_random.set_state(ss['env_rng'])

        self._epoch_paths = ss['_epoch_paths']
        self._num_steps_total = ss['_num_steps_total']
        self._num_paths_total = ss['_num_paths_total']


@ray.remote(num_cpus=1)
class RemoteMdpPathCollector(MdpPathCollector):

    def __init__(self,
                 domain_name, env_seed, goal,
                 policy_producer, max_num_epoch_paths_saved=None,
                 render=False, render_kwargs=None,
                 ):

        torch.set_num_threads(1)

        env = env_producer(domain_name, env_seed, goal)

        self._policy_producer = policy_producer

        super().__init__(env,
                         max_num_epoch_paths_saved=max_num_epoch_paths_saved,
                         render=render,
                         render_kwargs=render_kwargs,
                         )

    def async_collect_new_paths(self,
                                max_path_length,
                                num_steps,
                                discard_incomplete_paths,

                                deterministic_pol,
                                pol_state_dict):

        if deterministic_pol:
            policy = self._policy_producer(deterministic=True)
            policy.stochastic_policy.load_state_dict(pol_state_dict)

        else:
            policy = self._policy_producer()
            policy.load_state_dict(pol_state_dict)

        self.collect_new_paths(policy,
                               max_path_length, num_steps,
                               discard_incomplete_paths)

    def get_global_pkg_rng_state(self):
        return get_global_pkg_rng_state()

    def set_global_pkg_rng_state(self, state):
        set_global_pkg_rng_state(state)


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        optimistic_exploration=False,
        optimistic_exploration_kwargs={},
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:

        if not optimistic_exploration:
            a, agent_info = agent.get_action(o)
        else:
            a, agent_info = get_optimistic_exploration_action(
                o, **optimistic_exploration_kwargs)

        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


class tiMeSampler(object):
    def __init__(
        self, 
        env, 
        context_encoder, Qs, vae_decoder, perturbation_generator,
        vae_latent_dim,
        candidate_size=10,
        max_num_epoch_paths_saved=None):

        self.env = env

        self.context_encoder = context_encoder
        self.Qs = Qs
        self.vae_decoder = vae_decoder
        self.perturbation_generator = perturbation_generator
        self.nets = [self.context_encoder, self.Qs, self.vae_decoder, self.perturbation_generator]

        self.use_next_obs_in_context = True

        self.vae_latent_dim = vae_latent_dim
        self.candidate_size = candidate_size

        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
        self,
        max_path_length,
        num_steps,
        discard_incomplete_paths,
    ):

        self.context_encoder.clear_z()

        paths = []
        
        num_steps_collected = 0
        raw_context = deque()
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            # Sample MDP indentity
            self.context_encoder.sample_z()
            inferred_mdp = self.context_encoder.z

            path_length = 0
            observations = []
            actions = []
            rewards = []
            terminals = []
            agent_infos = []
            env_infos = []

            obs = self.env.reset()
            done = False
            
            while not done and path_length < max_path_length_this_loop:
                action = self.select_actions(np.array(obs), inferred_mdp)

                next_obs, reward, done, _ = self.env.step(action)

                if self.use_next_obs_in_context:
                    new_context = np.concatenate([
                        obs.reshape(1, -1), action.reshape(1, -1), next_obs.reshape(1, -1), np.array(reward).reshape(1, -1)
                    ], axis=1)
                else:
                    assert False

                observations.append(obs)
                rewards.append(reward)
                terminals.append(done)
                actions.append(action)
                agent_infos.append(-1)
                env_infos.append(-1)
                path_length += 1

                raw_context.append(new_context)
                obs = next_obs.copy()

            context = from_numpy(np.concatenate(raw_context, axis=0))[None]
            self.context_encoder.infer_posterior(context)
            
            actions = np.array(actions)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, 1)

            observations = np.array(observations)
            if len(observations.shape) == 1:
                observations = np.expand_dims(observations, 1)
                next_obs = np.array([next_obs])
            next_observations = np.vstack(
                (
                    observations[1:, :],
                    np.expand_dims(next_obs, 0)
                )
            )

            path = dict(
                observations=observations,
                actions=actions,
                rewards=np.array(rewards).reshape(-1, 1),
                next_observations=next_observations,
                terminals=np.array(terminals).reshape(-1, 1),
                agent_infos=agent_infos,
                env_infos=env_infos,
            )

            path_len = len(path['actions'])
            if (
                    # incomplete path
                    path_len != max_path_length and

                    # that did not end in a terminal state
                    not path['terminals'][-1] and

                    # and we should discard such path
                    discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

        return paths, inferred_mdp

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

    def to(self, device):
        for net in self.nets:
            net.to(device)

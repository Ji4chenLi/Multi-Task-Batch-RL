import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np
import ray

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_goals,
            wd_goals,
            ood_goals,
            replay_buffers,
            meta_batch_size=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_tasks=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1,
            num_exp_traj_eval=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            use_same_context=True,
            recurrent=False,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.train_goals = train_goals
        self.wd_goals = wd_goals
        self.ood_goals = ood_goals
        self.replay_buffers = replay_buffers
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.meta_batch_size = meta_batch_size
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.max_path_length = max_path_length
        self.discount = discount
        self.reward_scale = reward_scale
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.use_same_context = use_same_context
        self.recurrent=recurrent

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)

            # Sample train tasks and compute gradient updates on parameters.
            batch_idxes = np.random.randint(
                    0, len(self.train_goals), size=self.meta_batch_size
            )
            train_batch_obj_id = self.replay_buffers.sample_training_data(
                batch_idxes, self.use_same_context
            )
            for _ in range(self.num_train_steps_per_itr):
                train_raw_batch = ray.get(train_batch_obj_id)
                gt.stamp('sample_training_data', unique=False)

                batch_idxes = np.random.randint(
                    0, len(self.train_goals), size=self.meta_batch_size
                )
                # In this way, we can start the data sampling job for the
                # next training while doing training for the current loop.
                train_batch_obj_id = self.replay_buffers.sample_training_data(
                    batch_idxes, self.use_same_context
                )
                gt.stamp('set_up_sampling', unique=False)

                train_data = self.construct_training_batch(train_raw_batch)
                gt.stamp('construct_training_batch', unique=False)

                self._do_training(train_data)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()
            if it_ == self.num_iterations:
                logger.save_itr_params(it_, self.agent.get_snapshot())

    def construct_training_batch(self, raw_batch):
        ''' Construct training batch from raw batch'''
        state = np.concatenate([rb[0] for rb in raw_batch], axis=0)
        next_state = np.concatenate([rb[1] for rb in raw_batch], axis=0)
        actions = np.concatenate([rb[2] for rb in raw_batch], axis=0)
        rewards = np.concatenate([rb[3] for rb in raw_batch], axis=0)
        dones = np.concatenate([rb[4] for rb in raw_batch], axis=0)
        contexts = np.concatenate([rb[5] for rb in raw_batch], axis=0)

        return [state, next_state, actions, rewards, dones, contexts]

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass


    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)

            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample_training_data'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return True

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()


    ##### Snapshotting utils #####

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.agent,
        )
        if self.save_environment:
            data_to_save['env'] = self.env
        return data_to_save


    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        if epoch == self.num_iterations - 1:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, goal, epoch, run):
        self.env.set_goal(goal)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/evla_goal{}-epoch{}-run{}'.format(goal, epoch, run))

        return paths

    def _do_eval(self, goal_set, epoch):
        
        final_returns = []
        final_achieved = []
        for goal in goal_set:
            all_rets = []
            all_achieved = []
            for r in range(self.num_evals):
                paths = self.collect_paths(goal, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                all_achieved.append([eval_util.get_average_achieved([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            final_achieved.append(np.mean([a[-1] for a in all_achieved]))

        return final_returns, final_achieved

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        eval_util.dprint('evaluating on {} train tasks'.format(len(self.train_goals)))

        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_final_achieved = self._do_eval(self.train_goals, epoch)

        # Comment this line for walker-param
        # train_final_achieved_pair = [(train_final_achieved[i], goal) for i, goal in enumerate(self.train_goals)]
        train_final_achieved_pair = [(train_final_achieved[i], -1) for i, goal in enumerate(self.train_goals)]

        eval_util.dprint('train final achieved')
        eval_util.dprint(train_final_achieved_pair)

        ### WD tasks

        eval_util.dprint('evaluating on {} wd tasks'.format(len(self.wd_goals)))
        wd_final_returns, wd_final_achieved = self._do_eval(self.wd_goals, epoch)

        # Comment this line for walker-param
        # wd_final_achieved_pair = [(wd_final_achieved[i], goal) for i, goal in enumerate(self.wd_goals)]
        wd_final_achieved_pair = [(wd_final_achieved[i], -1) for i, goal in enumerate(self.wd_goals)]

        eval_util.dprint('WD test final achieved')
        eval_util.dprint(wd_final_achieved_pair)

        # ### OOD tasks

        # eval_util.dprint('evaluating on {} wd tasks'.format(len(self.ood_goals)))
        # ood_final_returns, ood_final_achieved = self._do_eval(self.ood_goals, epoch)

        # # Comment this line for walker-param
        # # ood_final_achieved_pair = [(ood_final_achieved[i], goal) for i, goal in enumerate(self.ood_goals)]
        # ood_final_achieved_pair = [(ood_final_achieved[i], -1) for i, goal in enumerate(self.ood_goals)]

        # eval_util.dprint('OOD test final achieved')
        # eval_util.dprint(ood_final_achieved_pair)

        # # save the final posterior
        # self.agent.log_diagnostics(self.eval_statistics)

        avg_train_return = np.mean(train_final_returns)
        avg_wd_return = np.mean(wd_final_returns)
        # avg_ood_return = np.mean(ood_final_returns)

        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_wd_tasks'] = avg_wd_return
        # self.eval_statistics['AverageReturn_all_ood_tasks'] = avg_ood_return

        self.eval_statistics['Return_all_train_tasks'] = train_final_returns
        self.eval_statistics['Return_all_wd_tasks'] = wd_final_returns
        # self.eval_statistics['Return_all_ood_tasks'] = ood_final_returns


        self.eval_statistics['Achieved_all_train_tasks'] = train_final_achieved_pair
        self.eval_statistics['Achieved_all_wd_tasks'] = wd_final_achieved_pair
        # self.eval_statistics['Achieved_all_ood_tasks'] = ood_final_achieved_pair

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None


        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self, train_data):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass


# default PEARL experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    domain='ant-dir',
    exp_mode='normal',
    seed=0,
    num_tasks=10,
    latent_size=5, # dimension of the latent context vector
    num_trans_context=64,
    in_mdp_batch_size=256,
    bcq_interactions=200,
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    algo_params=dict(
        meta_batch_size=16, # number of tasks to average the gradient across
        num_iterations=1000, # number of data sampling / training iterates
        num_train_steps_per_itr=100, # number of meta-gradient steps taken per iteration
        num_evals=1, # number of independent evals
        num_steps_per_eval=1000,  # number of transitions to eval on
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=True, # use next obs if it is useful in distinguishing tasks
        dump_eval_paths=False, # whether to save evaluation trajectories
        recurrent=False,
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    )
)




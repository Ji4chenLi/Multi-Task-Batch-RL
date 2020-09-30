# Focus especially on:
#   max_path_length
#   num_tasks
#   num_train_loops_per_epoch 
 
variant = dict(
    domain='humanoid-openai-dir',
    exp_mode='normal',
    seed=0,
    algo_params=dict(
        num_epochs=200,
        num_tasks=10,
        num_train_loops_per_epoch=20,
    ),
    std_threshold=10.0,
    num_network_ensemble=20,
    bcq_interactions=600,
    max_path_length=200,
    in_mdp_batch_size=128,
    candidate_size=10,
    P_hidden_sizes=[128],
    base_log_dir='data_reward_predictions',
    start=0,
    end=1,
)
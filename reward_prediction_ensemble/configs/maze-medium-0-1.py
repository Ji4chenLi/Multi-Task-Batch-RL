# Focus especially on:
#   max_path_length
#   num_tasks
#   num_train_loops_per_epoch 
 
variant = dict(
    domain='maze-medium',
    exp_mode='normal',
    seed=0,
    algo_params=dict(
        num_epochs=200,
        num_tasks=10,
        num_train_loops_per_epoch=20,
    ),
    std_threshold=0.015,
    num_network_ensemble=20,
    bcq_interactions=100,
    max_path_length=600,
    in_mdp_batch_size=128,
    candidate_size=10,
    P_hidden_sizes=[128, 128],
    base_log_dir='data_reward_predictions',
    start=0,
    end=1,
)
# Focus especially on:
#   max_path_length
#   num_tasks
#   num_train_loops_per_epoch 
 
num_tasks = 10

variant = dict(
    domain='ant-dir',
    exp_mode='normal',
    seed=0,
    max_path_length=200,
    num_trans_context=128,
    in_mdp_batch_size=128,
    num_evals=5,
    bcq_interactions=200,
    policy_params=dict(
        vae_latent_dim_multiplicity=2, 
        target_q_coef=0.75,
        actor_hid_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], 
        critic_hid_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], 
        vae_e_hid_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024], 
        vae_d_hid_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024],
        encoder_latent_dim=20, 
    ),
    algo_params=dict(
        num_epochs=1000,
        num_train_loops_per_epoch=100,
        num_tasks=num_tasks,
    ),
    base_log_dir='./data_without_model_prediction',
)

assert num_tasks == 10

assert variant['num_trans_context'] == 128

assert len(variant['policy_params']['actor_hid_sizes']) == 8
assert len(variant['policy_params']['critic_hid_sizes']) == 9
assert len(variant['policy_params']['vae_e_hid_sizes']) == 7
assert len(variant['policy_params']['vae_d_hid_sizes']) == 7
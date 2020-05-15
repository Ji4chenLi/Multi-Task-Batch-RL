# Focus especially on:
#   max_path_length
#   num_tasks
#   num_train_loops_per_epoch 
 
num_tasks = 10

variant = dict(
    domain='halfcheetah-vel',
    exp_mode='hard',
    seed=0,
    max_path_length=1000,
    bcq_interactions=60,
    num_evals=5,
    use_next_obs_in_context=True,
    is_combine=False,
    algo_params=dict(
        num_epochs=1000,
        num_train_loops_per_epoch=100,
        num_tasks=num_tasks,
    ),
    latent_dim=20,
    triplet_margin=2.0,
    std_threshold=0.025,
    in_mdp_batch_size=128,
    candidate_size=10,
    num_trans_context=128,
    num_network_ensemble=20,
    Qs_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    vae_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024],
    perturbation_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    base_log_dir='./data_without_model_prediction',
)

assert variant['num_trans_context'] == 128
assert variant['triplet_margin'] == 2.0
assert variant['std_threshold'] == 0.025
assert num_tasks == 10
assert variant['is_combine'] == False

assert len(variant['Qs_hidden_sizes']) == 9
assert len(variant['vae_hidden_sizes']) == 7
assert len(variant['perturbation_hidden_sizes']) == 8

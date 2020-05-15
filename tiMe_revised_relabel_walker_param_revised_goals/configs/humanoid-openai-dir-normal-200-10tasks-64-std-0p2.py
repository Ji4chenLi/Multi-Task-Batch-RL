# Focus especially on:
#   max_path_length
#   num_tasks
#   num_train_loops_per_epoch 
 
num_tasks = 10

variant = dict(
    domain='humanoid-openai-dir',
    exp_mode='normal',
    seed=0,
    max_path_length=200,
    bcq_interactions=600,
    num_evals=5,
    use_next_obs_in_context=True,
    is_combine=True,
    algo_params=dict(
        num_epochs=1000,
        num_train_loops_per_epoch=100,
        num_tasks=num_tasks,
    ),
    latent_dim=20,
    std_threshold=0.2,
    in_mdp_batch_size=128,
    candidate_size=10,
    num_trans_context=64,
    num_network_ensemble=20,
    Qs_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    vae_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024],
    perturbation_hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    base_log_dir='./data_without_model_prediction',
)

assert variant['std_threshold'] == 0.2
assert variant['max_path_length'] == 200
assert num_tasks == 10

assert len(variant['Qs_hidden_sizes']) == 9
assert len(variant['vae_hidden_sizes']) == 7
assert len(variant['perturbation_hidden_sizes']) == 8

if variant['is_combine']:
    assert variant['num_trans_context'] == 64
else:
    assert variant['num_trans_context'] == 128
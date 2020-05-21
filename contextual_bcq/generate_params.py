
# ------------------------------------------------------------------------------

# from itertools import product
# import pickle


# actor = [[512] * i for i in range(3, 6)]

# critic = [[512] * i for i in range(3, 6)]

# vae_e = [[750] * i for i in range(2, 5)]

# vae_d = [[750] * i for i in range(3, 6)]


# all_hyperparams = list(product(actor, critic, vae_e, vae_d))


# with open('hyperparams_sat_feb_15.pkl', 'wb+') as f:
#     pickle.dump(all_hyperparams, f)

# ------------------------------------------------------------------------------

# from itertools import product
# import pickle


# actor = [[512] * i for i in range(3, 6)]

# critic = [[512] * i for i in range(3, 6)]

# vae_e = [[750] * i for i in range(2, 5)]

# vae_d = [[750] * i for i in range(3, 6)]


# networks_hyperparams_all = list(product(actor, critic, vae_e, vae_d))
# used_network_idx = [1, 3, 15, 12]
# networks_hyperparams_used = [networks_hyperparams_all[i] for i in used_network_idx]

# latent_dim = [8*4, 8*8]
# target_q_coefficient = [0.75, 0.9, 1.0]

# all_hyperparams = list(product(networks_hyperparams_used, latent_dim, target_q_coefficient))

# with open('hyperparams_sat_feb_15_afternoon.pkl', 'wb+') as f:
#     pickle.dump(all_hyperparams, f)
# ------------------------------------------------------------------------------
# import pickle


# actor = [512, 512, 512]

# critic = [512, 512, 512]

# vae_e = [750, 750, 750]

# vae_d = [750, 750, 750]

# latent_dim = 32

# target_q_coefficient = 0.9

# networks_hyperparams = [actor, critic, vae_e, vae_d]

# all_hyperparams = [[networks_hyperparams, latent_dim, target_q_coefficient]]

# print(all_hyperparams[0])

# with open('hyperparams_sat_feb_15_eval_goals.pkl', 'wb+') as f:
#     pickle.dump(all_hyperparams, f)


# ------------------------------------------------------------------------------

# import pickle


# actor = [400, 300]

# critic = [400, 300]

# vae_e = [750, 750]

# vae_d = [750, 750]

# latent_dim_multiplicity = 2

# target_q_coefficient = 0.75

# networks_hyperparams = [actor, critic, vae_e, vae_d]

# all_hyperparams = [[networks_hyperparams, latent_dim_multiplicity, target_q_coefficient]]

# print(all_hyperparams[0])

# with open('hyperparams_BCQ_default.pkl', 'wb+') as f:
#     pickle.dump(all_hyperparams, f)


# ------------------------------------------------------------------------------

# import pickle


# actor = [512, 512, 512, 512, 512, 512, 512]

# critic = [512, 512, 512, 512, 512, 512, 512]

# vae_e = [512, 512, 512, 512, 512, 512, 512]

# vae_d = [512, 512, 512, 512, 512, 512, 512]

# latent_dim_multiplicity = 2

# target_q_coefficient = 0.75

# networks_hyperparams = [actor, critic, vae_e, vae_d]

# all_hyperparams = [[networks_hyperparams, latent_dim_multiplicity, target_q_coefficient]]

# print(all_hyperparams[0])

# with open('hyperparams_bcq_encoder_feb_18_smaller.pkl', 'wb+') as f:
#     pickle.dump(all_hyperparams, f)

# ---------------------------------------------------------------------------------

import pickle


actor = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]

critic = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]

vae_e = [1024, 1024, 1024, 1024, 1024, 1024, 1024]

vae_d = [1024, 1024, 1024, 1024, 1024, 1024, 1024]

latent_dim_multiplicity = 2

target_q_coefficient = 0.75

networks_hyperparams = [actor, critic, vae_e, vae_d]

all_hyperparams = [[networks_hyperparams, latent_dim_multiplicity, target_q_coefficient]]

print(all_hyperparams[0])

with open('hyperparams_bcq_encoder_match_tiMe.pkl', 'wb+') as f:
    pickle.dump(all_hyperparams, f)
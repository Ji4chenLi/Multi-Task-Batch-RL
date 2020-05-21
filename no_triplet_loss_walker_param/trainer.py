import ray

import gtimer as gt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict, deque

import utils.pytorch_util as ptu
from utils.pytorch_util import np_to_pytorch_batch


def assert_pt(a, b, msg):

    assert torch.all(torch.eq(a, b)) == 1, msg


class SuperQTrainer(object):
    def __init__(
            self,
            nets,
            num_network_ensemble,
            reward_ensemble_predictor,
            transition_ensemble_predictor,
            bcq_policies,
            is_combine,
            general_lr=3e-4,
            kl_lambda=.1,
            reward_std_threshold=.1,
            next_obs_std_threshold=.1,
            optimizer_class=optim.Adam,

    ):
        super().__init__()

        self.bcq_polices = bcq_policies
        self.combined_bcq_policies = self.combine_bcq_policies(bcq_policies)

        self.num_network_ensemble = num_network_ensemble

        self.reward_ensemble_predictor = reward_ensemble_predictor
        self.transition_ensemble_predictor = transition_ensemble_predictor
        
        self.context_encoder, self.Qs, self.vae_decoder, self.perturbation_generator = nets

        self.context_encoder_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=general_lr
        )
        self.Qs_optimizer = optimizer_class(
            self.Qs.parameters(),
            lr=general_lr
        )
        self.vae_decoder_optimizer = optimizer_class(
            self.vae_decoder.parameters(),
            lr=general_lr
        )
        self.perturbation_generator_optimizer = optimizer_class(
            self.perturbation_generator.parameters(),
            lr=general_lr
        )

        self.is_combine = is_combine
        
        self.reward_std_threshold = reward_std_threshold
        self.next_obs_std_threshold = next_obs_std_threshold

        self.kl_lambda = kl_lambda

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    # @profile
    def train(self, batch, batch_idxes):

        """
        Unpack data from the batch
        """
        obs = batch['obs']
        actions = batch['actions']
        contexts = batch['contexts']

        num_tasks = batch_idxes.shape[0]

        gt.stamp('unpack_data_from_the_batch', unique=False)

        # Get the in_mdp_batch_size
        obs_dim = obs.shape[1]
        action_dim = actions.shape[1]
        in_mdp_batch_size = obs.shape[0] // batch_idxes.shape[0]
        num_trans_context = contexts.shape[0] // batch_idxes.shape[0]

        """
        Relabel the context batches for each training task
        """
        with torch.no_grad():
            contexts_obs_actions = contexts[:, :obs_dim + action_dim]

            manual_batched_rewards = self.reward_ensemble_predictor.forward_mul_device(contexts_obs_actions)
            
            relabeled_rewards = manual_batched_rewards.reshape(
                num_tasks, self.num_network_ensemble, contexts.shape[0]
            )

            gt.stamp('relabel_ensemble', unique=False)

            manual_batched_next_obs = self.transition_ensemble_predictor.forward_mul_device(contexts_obs_actions)

            relabeled_next_obs = manual_batched_next_obs.reshape(
                num_tasks, self.num_network_ensemble, contexts.shape[0], obs_dim
            )

            gt.stamp('transition_ensemble_forward', unique=False)

            relabeled_rewards_mean = torch.mean(relabeled_rewards, dim=1).squeeze()
            relabeled_rewards_std = torch.std(relabeled_rewards, dim=1).squeeze()

            relabeled_next_obs_mean = torch.mean(relabeled_next_obs, dim=1)

            relabeled_next_obs_std = torch.std(relabeled_next_obs, dim=1)
            relabeled_next_obs_std = torch.mean(relabeled_next_obs_std, dim=-1)

            # Replace the predicted reward with ground truth reward for transitions
            # with ground truth reward inside the batch
            for i in range(num_tasks):
                relabeled_rewards_mean[i, i*num_trans_context: (i+1)*num_trans_context] \
                    = contexts[i*num_trans_context: (i+1)*num_trans_context, -1]

                relabeled_next_obs_mean[i, i*num_trans_context: (i+1)*num_trans_context, :] \
                    = contexts[i*num_trans_context: (i+1)*num_trans_context, obs_dim + action_dim : -1]

                if self.is_combine:
                    # Set the number to be larger than the self.std_threshold, so that
                    # they will initially be filtered out when producing the mask,
                    # which is conducive to the sampling.
                    relabeled_rewards_std[i, i*num_trans_context: (i+1)*num_trans_context] = self.reward_std_threshold + 1.0
                    relabeled_next_obs_std[i, i*num_trans_context: (i+1)*num_trans_context] = self.next_obs_std_threshold + 1.0
                else:
                    relabeled_rewards_std[i, i*num_trans_context: (i+1)*num_trans_context] = 0.0
                    relabeled_next_obs_std[i, i*num_trans_context: (i+1)*num_trans_context] = 0.0

            mask_reward = relabeled_rewards_std < self.reward_std_threshold
            mask_reward = mask_reward.type(torch.float)

            mask_next_obs = relabeled_next_obs_std < self.next_obs_std_threshold
            mask_next_obs = mask_next_obs.type(torch.float)

            mask = mask_reward * mask_next_obs
            mask = mask.type(torch.uint8)

            num_context_candidate_each_task = torch.sum(mask, dim=1)

            mask_list = []

            for i in range(num_tasks):

                assert mask[i].dim() == 1

                mask_nonzero = torch.nonzero(mask[i])
                mask_nonzero = mask_nonzero.flatten()

                mask_i = ptu.zeros_like(mask[i], dtype=torch.uint8) 

                assert num_context_candidate_each_task[i].item() == mask_nonzero.shape[0]

                np_ind = np.random.choice(
                    mask_nonzero.shape[0], num_trans_context, replace=False)

                ind = mask_nonzero[np_ind]

                mask_i[ind] = 1

                if self.is_combine:
                    # Combine the additional relabeledcontext transitions with
                    # the original context transitions with ground-truth rewards
                    mask_i[i*num_trans_context: (i+1)*num_trans_context] = 1

                    assert torch.sum(mask_i).item() == 2 * num_trans_context
                else:
                    assert torch.sum(mask_i).item() == num_trans_context

                mask_list.append(mask_i)

            mask = torch.cat(mask_list)
            mask = mask.type(torch.uint8)

            repeated_contexts = contexts.repeat(num_tasks, 1)
            context_without_next_obs_rewards = repeated_contexts[:, : obs_dim + action_dim]

            assert context_without_next_obs_rewards.shape[0] == relabeled_rewards_mean.reshape(-1, 1).shape[0]
            assert context_without_next_obs_rewards.shape[0] == relabeled_next_obs_mean.reshape(-1, obs_dim).shape[0]

            context_without_next_obs_rewards = context_without_next_obs_rewards[mask]

            context_next_obs = relabeled_next_obs_mean.reshape(-1, obs_dim)[mask]

            context_rewards = relabeled_rewards_mean.reshape(-1, 1)[mask]

            fast_contexts = torch.cat((context_without_next_obs_rewards, context_next_obs, context_rewards), dim=1)

            fast_contexts = fast_contexts.reshape(num_tasks, -1, contexts.shape[-1])

        gt.stamp('relabel_context_transitions', unique=False)
        
        """
        Infer the context variables
        """
        # inferred_mdps = self.context_encoder(new_contexts)
        inferred_mdps = self.context_encoder(fast_contexts)
        inferred_mdps = torch.repeat_interleave(inferred_mdps, in_mdp_batch_size, dim=0) 
        
        gt.stamp('infer_mdps', unique=False)

        with torch.no_grad():
            # Sample z for each state
            z = self.bcq_polices[0].vae.sample_z(obs).to(ptu.device)

            # Each item in critic_weights is a list that has device count entries
            # each entry in the critic_weights[i] is a list that has num layer entries
            # each entry in the critic_weights[i][j] is a tensor of dim (num tasks // device count, layer input size, layer out size)
            # Similarly to the other weights and biases
            critic_weights, critic_biases, vae_weights, vae_biases, actor_weights, actor_biases = self.combined_bcq_policies

            # CRITIC
            obs_reshaped = obs.reshape(len(batch_idxes), in_mdp_batch_size, -1)
            acs_reshaped = actions.reshape(len(batch_idxes), in_mdp_batch_size, -1)

            obs_acs_reshaped = torch.cat((obs_reshaped, acs_reshaped), dim=-1)

            target_q = batch_bcq(obs_acs_reshaped, critic_weights, critic_biases)
            target_q = target_q.reshape(-1)

            # VAE
            z_reshaped = z.reshape(len(batch_idxes), in_mdp_batch_size, -1)
            obs_z_reshaped = torch.cat((obs_reshaped, z_reshaped), dim=-1)

            tc = batch_bcq(obs_z_reshaped, vae_weights, vae_biases)
            tc = self.bcq_polices[0].vae.max_action * torch.tanh(tc)
            target_candidates = tc.reshape(-1, tc.shape[-1])

            # PERTURBATION
            tc_reshaped = target_candidates.reshape(len(batch_idxes), in_mdp_batch_size, -1)

            obs_tc_reshaped = torch.cat((obs_reshaped, tc_reshaped), dim=-1)

            tp = batch_bcq(obs_tc_reshaped, actor_weights, actor_biases)
            tp = self.bcq_polices[0].actor.max_action * torch.tanh(tp)
            target_perturbations = tp.reshape(-1, tp.shape[-1])

        gt.stamp('get_the_targets', unique=False)
        """
        Obtain the KL loss
        """
        
        # KL constraint on z if probabilistic
        self.context_encoder_optimizer.zero_grad()

        kl_div = self.context_encoder.compute_kl_div()

        kl_loss_each_task = self.kl_lambda * torch.sum(kl_div, dim=1)

        kl_loss = torch.sum(kl_loss_each_task)

        gt.stamp('get_kl_loss', unique=False)
        """
        Obtain the Q-function loss
        """
        self.Qs_optimizer.zero_grad()

        pred_q = self.Qs(obs, actions, inferred_mdps)
        pred_q = torch.squeeze(pred_q)

        qf_loss_each_task = (pred_q - target_q) ** 2
        qf_loss_each_task = qf_loss_each_task.reshape(num_tasks, -1)
        qf_loss_each_task = torch.mean(qf_loss_each_task, dim=1)

        qf_loss = torch.mean(qf_loss_each_task)

        gt.stamp('get_qf_loss', unique=False)

        (kl_loss + qf_loss).backward()

        gt.stamp('get_kl_qf_gradient', unique=False)

        self.Qs_optimizer.step()
        self.context_encoder_optimizer.step()

        gt.stamp('update_Qs_encoder', unique=False)

        """
        Obtain the candidate action and perturbation loss
        """
        
        self.vae_decoder_optimizer.zero_grad()
        self.perturbation_generator_optimizer.zero_grad()

        pred_candidates = self.vae_decoder(
            obs, z, inferred_mdps.detach()
        )
        pred_perturbations = self.perturbation_generator(
            obs, target_candidates, inferred_mdps.detach()
        )

        candidate_loss_each_task = (pred_candidates - target_candidates) ** 2

        # averaging over action dimension
        candidate_loss_each_task = torch.mean(candidate_loss_each_task, dim=1)
        candidate_loss_each_task = candidate_loss_each_task.reshape(num_tasks, in_mdp_batch_size)

        # average over action in each task
        candidate_loss_each_task = torch.mean(candidate_loss_each_task, dim=1)

        candidate_loss = torch.mean(candidate_loss_each_task)

        perturbation_loss_each_task = (pred_perturbations - target_perturbations) ** 2

        # average over action dimension
        perturbation_loss_each_task = torch.mean(perturbation_loss_each_task, dim=1)
        perturbation_loss_each_task = perturbation_loss_each_task.reshape(num_tasks, in_mdp_batch_size)

        # average over action in each task
        perturbation_loss_each_task = torch.mean(perturbation_loss_each_task, dim=1)

        perturbation_loss = torch.mean(perturbation_loss_each_task)

        gt.stamp('get_candidate_and_perturbation_loss', unique=False)

        (candidate_loss + perturbation_loss).backward()

        gt.stamp('get_candidate_and_perturbation_gradient', unique=False)

        self.vae_decoder_optimizer.step()
        self.perturbation_generator_optimizer.step()

        gt.stamp('update_vae_perturbation', unique=False)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['qf_loss'] = np.mean(
                ptu.get_numpy(qf_loss)
            )
            self.eval_statistics['qf_loss_each_task'] = ptu.get_numpy(qf_loss_each_task)
                
            self.eval_statistics['kl_loss'] = np.mean(
                ptu.get_numpy(kl_loss)
            )
            self.eval_statistics['kl_loss_each_task'] = ptu.get_numpy(kl_loss_each_task)

            self.eval_statistics['candidate_loss'] = np.mean(
                ptu.get_numpy(candidate_loss)
            )
            self.eval_statistics['candidate_loss_each_task'] = ptu.get_numpy(candidate_loss_each_task)

            self.eval_statistics['perturbation_loss'] = np.mean(
                ptu.get_numpy(perturbation_loss)
            )
            self.eval_statistics['perturbation_loss_each_task'] = ptu.get_numpy(perturbation_loss_each_task)

            self.eval_statistics['num_context_candidate_each_task'] = num_context_candidate_each_task
            
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.context_encoder.mlp_encoder, self.Qs, self.vae_decoder, self.perturbation_generator]

    def get_snapshot(self):
        return dict(
            context_encoder_state_dict=self.context_encoder.state_dict(),
            Qs_state_dict=self.Qs.state_dict(),
            vae_decoder_state_dict=self.vae_decoder.state_dict(),
            perturbation_generator_dict=self.perturbation_generator.state_dict(),

            context_encoder_optimizer_state_dict=self.context_encoder_optimizer.state_dict(),
            Qs_optimizer_state_dict=self.Qs_optimizer.state_dict(),
            vae_decoder_optimizer_state_dict=self.vae_decoder_optimizer.state_dict(),
            perturbation_generator_optimizer_state_dict=self.perturbation_generator_optimizer.state_dict(),

            eval_statistics=self.eval_statistics,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
        )

    def restore_from_snapshot(self, ss):
        ss = ss['trainer']
        self.context_encoder.load_state_dict(ss['context_encoder_state_dict'])
        self.context_encoder.to(ptu.device)
        self.Qs.load_state_dict(ss['Qs_state_dict'])
        self.Qs.to(ptu.device)
        self.vae_decoder.load_state_dict(ss['vae_decoder_state_dict'])
        self.vae_decoder.to(ptu.device)
        self.perturbation_generator.load_state_dict(ss['perturbation_generator_dict'])
        self.perturbation_generator.to(ptu.device)

        self.eval_statistics = ss['eval_statistics']
        self._need_to_update_eval_statistics = ss['_need_to_update_eval_statistics']

    def combine_bcq_policies(self, bcq_polices):

        device_count = torch.cuda.device_count()

        idxes = torch.arange(len(bcq_polices))
        chunked_idxes = torch.chunk(idxes, device_count, dim=0)

        chunked_policies = []

        for chunk in chunked_idxes:

            tmp = []

            for idx in chunk:
                tmp.append(bcq_polices[idx.int().item()])

            chunked_policies.append(tmp)

        critic_weights_mul_dev = []
        critic_biases_mul_dev = []

        vae_weights_mul_dev = []
        vae_biases_mul_dev = []

        actor_weights_mul_dev = []
        actor_biases_mul_dev = []

        for device_i in range(device_count):

            device = torch.device(f'cuda:{device_i}')

            one_chunk_pol = chunked_policies[device_i]

            # CRITIC
            critic_weight_names = ['l1_ipt.weight', 'l1_linears.0.weight', 'l1_out.weight']
            critic_bias_names = ['l1_ipt.bias', 'l1_linears.0.bias', 'l1_out.bias']

            # critic params is a list
            # each item is a tensor of size
            # (num_tasks, layer input size, layer output size)
            critic_state_dicts = [p.critic.state_dict() for p in one_chunk_pol]

            critic_weights = nets_to_ensemble(critic_state_dicts, critic_weight_names, device, 'weight')
            critic_biases = nets_to_ensemble(critic_state_dicts, critic_bias_names, device, 'bias') 

            critic_weights_mul_dev.append(critic_weights)
            critic_biases_mul_dev.append(critic_biases)

            # VAE
            vae_weight_names = [ 'd_ipt.weight', 'd_linears.0.weight','d_out.weight' ]
            vae_bias_names = ['d_ipt.bias',  'd_linears.0.bias',  'd_out.bias' ]

            state_dicts = [p.vae.state_dict() for p in one_chunk_pol]
            vae_weights = nets_to_ensemble(state_dicts, vae_weight_names, device, 'weight')
            vae_biases = nets_to_ensemble(state_dicts, vae_bias_names, device, 'bias')

            vae_weights_mul_dev.append(vae_weights)
            vae_biases_mul_dev.append(vae_biases)

            # ACTOR
            actor_weight_names = ['l_ipt.weight', 'linears.0.weight', 'l_out.weight']
            actor_bias_names = ['l_ipt.bias', 'linears.0.bias', 'l_out.bias']

            state_dicts = [p.actor.state_dict() for p in one_chunk_pol]

            actor_weights = nets_to_ensemble(state_dicts, actor_weight_names, device, 'weight')
            actor_biases = nets_to_ensemble(state_dicts, actor_bias_names, device, 'bias')

            actor_weights_mul_dev.append(actor_weights)
            actor_biases_mul_dev.append(actor_biases)

        return critic_weights_mul_dev, critic_biases_mul_dev, \
            vae_weights_mul_dev, vae_biases_mul_dev, \
                actor_weights_mul_dev, actor_biases_mul_dev


def batch_bcq(inputs, weights, biases):
    '''Receive as inputs a 3d tensor [num task, in mdp batch size, input dim]
    
    weights and biases are list.
    
    vae decode and actor get perturbation operations require 
    additional post processing outside this function'''

    dc = torch.cuda.device_count()

    chunked_inputs = torch.chunk(inputs, dc, dim=0)   

    results = []

    for i, c_i in enumerate(chunked_inputs):

        x = c_i.to(torch.device(f'cuda:{i}'))

        device_weights = weights[i]
        device_biases = biases[i]

        for layer_idx in range(len(device_weights)):

            w = device_weights[layer_idx]
            b = device_biases[layer_idx]

            x = torch.bmm(x, w) + b

            # If not last layer
            if layer_idx < len(device_weights) - 1:
                x = F.relu(x)

        results.append(x)

    results = torch.cat(results, dim=0)

    return results


def nets_to_ensemble(state_dicts, names, device, transform=None):
    all_layers_val = []

    for name in names:

        one_layer_val = []

        for sd in state_dicts:

            v = sd[name]
            
            if transform == 'weight':
                # .t() because we want the input size to come first 
                v = v.t()

            elif transform == 'bias':
                v = v.reshape(1, -1)

            else:
                assert False

            # add None to make concat below works
            v = v[None]

            one_layer_val.append(v)

        one_layer_val = torch.cat(one_layer_val, dim=0)
        one_layer_val = one_layer_val.to(device)

        all_layers_val.append(one_layer_val)

    return all_layers_val



# Fast version of relabeling rewards for context, not 100% suure its correctness.

# for i in range(num_tasks):
#     relabeled_rewards[i, :, i*num_trans_context: (i+1)*num_trans_context] = contexts[i*num_trans_context: (i+1)*num_trans_context, -1]

    # relabeled_rewards_mean = torch.mean(relabeled_rewards, dim=1).squeeze()
    # relabeled_rewards_std = torch.std(relabeled_rewards, dim=1).squeeze()

    # # order = np.arange(contexts.shape[0])
    # # oder_tensor[mask[i]]

    # mask = ptu.get_numpy(relabeled_rewards_std < self.std_threshold)
    # num_context_candidate_each_task = np.sum(mask, axis=1)

    # for i in range(num_tasks):
    #     ind = np.random.choice(
    #         np.where(mask[i] == 1)[0], int(num_context_candidate_each_task[i] - num_trans_context), replace=False
    #     )
    #     mask[i, ind] = 0

    # mask = ptu.from_numpy(mask) > 0

    # context_without_rewards = contexts.repeat(num_tasks, 1)[:, :-1][mask.flatten()]

    # context_rewards = relabeled_rewards_mean.reshape(-1, 1)[mask.flatten()]

    # contexts = torch.cat((context_without_rewards, context_rewards), dim=1)

    # contexts = contexts.reshape(num_tasks, num_trans_context, -1)


# April 7 2020
# Original sequential code to relabel the contet transition
# based on the ensemble std
# contexts_obs_actions = contexts[:, :obs_dim + action_dim]

# relabeled_rewards = self.ensemble_predictor(contexts_obs_actions).reshape(
#     num_tasks, self.num_network_ensemble, contexts.shape[0]
# )

# relabeled_rewards_mean = torch.mean(relabeled_rewards, dim=1).squeeze()
# relabeled_rewards_std = torch.std(relabeled_rewards, dim=1).squeeze()

# new_contexts = []
# num_context_candidate_each_task = []
# index_list = []
# mask_list = []

# contexts_mids = []
# add_contexts = []

# for i in batch_idxes:
#     mask = torch.cat((
#         relabeled_rewards_std[i, :i*num_trans_context], relabeled_rewards_std[i, (i+1)*num_trans_context:]
#     )) < self.std_threshold

#     mask_list.append(mask)

#     num_context_candidate_each_task.append((torch.sum(mask) + num_trans_context).item())

#     # contexts: 
#     # number of transitions inside one context (num_trans_context) * number of task, 
#     # dimension of one context transition
#     add_context_without_rewards = torch.cat((
#         contexts[:i*num_trans_context, :-1], contexts[(i+1)*num_trans_context:, :-1]
#     ), dim=0)[mask]

#     add_context_rewards = torch.cat((
#         relabeled_rewards_mean[i, :i*num_trans_context], relabeled_rewards_mean[i, (i+1)*num_trans_context:]
#     ), dim=0)[mask].reshape(-1, 1)

#     add_context = torch.cat((add_context_without_rewards, add_context_rewards), dim=1)

#     contexts_mids.append(contexts[i*num_trans_context:(i+1)*num_trans_context])

#     add_contexts.append(add_context)

#     new_context = torch.cat(
#         (contexts[i*num_trans_context:(i+1)*num_trans_context], 
#         add_context), dim=0)
    
#     ind = np.random.choice(new_context.shape[0], num_trans_context, replace=False)
#     new_context = new_context[ind]

#     index_list.append(ind)

#     new_contexts.append(new_context[None])

# new_contexts = torch.cat(new_contexts, dim=0)


# April 7 2020 
# Sequential version of the batch version to relabel the transition
# using the ensemble std
            # ----------------------------------------------------------------------------------------------------------

            # relabeled_rewards = self.ensemble_predictor(contexts_obs_actions).reshape(
            #     num_tasks, self.num_network_ensemble, contexts.shape[0]
            # )

            # relabeled_rewards_mean = torch.mean(relabeled_rewards, dim=1).squeeze()
            # relabeled_rewards_std = torch.std(relabeled_rewards, dim=1).squeeze()

            # # Replace the predicted reward with ground truth reward for transitions
            # # with ground truth reward inside the batch
            # for i in range(num_tasks):
            #     relabeled_rewards_mean[i, i*num_trans_context: (i+1)*num_trans_context] \
            #         = contexts[i*num_trans_context: (i+1)*num_trans_context, -1]

            #     relabeled_rewards_std[i, i*num_trans_context: (i+1)*num_trans_context] = 0

            # fast_seq_contexts = []

            # index_list = [] # Uncoment this when compared with the batch version
            # for i in batch_idxes:

            #     mask = relabeled_rewards_std[i] < self.std_threshold

            #     new_mask = torch.cat((
            #         mask[:i*num_trans_context], mask[(i+1)*num_trans_context:]
            #     ))

            #     # assert torch.all(torch.eq(mask_list[i], new_mask)).item() == 1

            #     non_zero_pos = mask.nonzero().flatten()

            #     mid_ind = []

            #     for j, pos in enumerate(non_zero_pos):
            #         if pos >= i * num_trans_context and pos < (i+1) * num_trans_context:
            #             mid_ind.append(j)

            #     # num_context_candidate_each_task.append(torch.sum(mask).item())

            #     context_without_rewards = contexts[:, :-1][mask]

            #     context_rewards = relabeled_rewards_mean[i][mask].reshape(-1, 1)

            #     context = torch.cat((context_without_rewards, context_rewards), dim=1)

            #     # context_before = context[:mid_ind[0]]
            #     # context_mid = context[mid_ind]
            #     # context_after = context[mid_ind[-1] + 1:]

            #     # context_before_after = torch.cat([context_before, context_after], dim=0)

            #     # for num_trans in range(contexts_mids[i].shape[0]):
            #     #     assert_pt(
            #     #         contexts_mids[i][num_trans],
            #     #         context_mid[num_trans],
            #     #         f'{num_tasks}, {contexts_mids[i][num_trans]}, {context_mid[num_trans]}'
            #     #     )    

            #     # assert_pt(
            #     #     contexts_mids[i], context_mid, i
            #     # )

            #     # assert_pt(
            #     #     add_contexts[i],
            #     #     context_before_after, i
            #     # )

            #     # context = torch.cat([
            #     #     context_mid, context_before_after
            #     # ], dim=0)

            #     # ind = index_list[i]

            #     ind = np.random.choice(context.shape[0], num_trans_context, replace=False)
            #     ind.sort()
            #     index_list.append(ind)

            #     context = context[ind]

            #     fast_seq_contexts.append(context[None])

            # fast_seq_contexts = torch.cat(fast_seq_contexts, dim=0)

            # # assert_pt(new_contexts, fast_seq_contexts, 'compare new contexts and fast seg context')

            # # print([F.mse_loss(new_contexts[i], fast_seq_contexts[i]) for i in range(num_tasks)])


# April 7 2020
# Assert statement for relabel transition
# using ensemble std
            # assert_pt(fast_contexts, fast_seq_contexts, 'compare context')
            # # print([torch.all(torch.eq(fast_contexts[i], fast_seq_contexts[i])).item() for i in range(num_tasks)])

            # # print([F.mse_loss(fast_contexts[i], fast_seq_contexts[i]).item() for i in range(num_tasks)])

            # # _, new_contexts_mdp_mu, new_contexts_mdp_var = self.context_encoder(new_contexts, return_prob_params=True)
            # _, fast_seq_contexts_mdp_mu, fast_seq_contexts_mdp_var = self.context_encoder(fast_seq_contexts, return_prob_params=True)
            # _, fast_contexts_mdp_mu, fast_contexts_mdp_var = self.context_encoder(fast_contexts, return_prob_params=True)

            # # assert_pt(new_contexts_mdp_mu, fast_seq_contexts_mdp_mu, 'a')
            # assert_pt(fast_seq_contexts_mdp_mu, fast_contexts_mdp_mu, 'b')

            # # assert_pt(new_contexts_mdp_var, fast_seq_contexts_mdp_var, 'c')
            # assert_pt(fast_seq_contexts_mdp_var, fast_contexts_mdp_var, 'd')

            # # exit()

# April 7 2020
#  numpy code to get mask for relabel transition batch version
            # ---------------------------
            # mask = ptu.get_numpy(relabeled_rewards_std < self.std_threshold)

            # num_context_candidate_each_task = np.sum(mask, axis=1)

            # mask_list = []
            # ind_list = []
            # for i in range(num_tasks):

            #     assert mask[i].ndim == 1

            #     mask_nonzero = mask[i].nonzero()
            #     nonzero_0_idx = mask_nonzero[0]

            #     nozero_pos = nonzero_0_idx.flatten()

            #     mask_i = np.zeros(mask[i].shape).astype(np.int)

            #     assert num_context_candidate_each_task[i] == nozero_pos.shape[0] 

            #     ind = np.random.choice(num_context_candidate_each_task[i], num_trans_context, replace=False)
            #     ind = nozero_pos[ind]

            #     ind_list.append(ind)
            #     mask_i[ind] = 1

            #     mask_list.append(mask_i)

            # mask = ptu.from_numpy(np.concatenate(mask_list))
            # mask = mask.type(torch.uint8)

            # ---------------------------

# April 7 2020
# code to run ensemble given a batch of obs action on ONE gpu
            # relabeled_rewards_before_reshape = self.ensemble_predictor(contexts_obs_actions)
            # relabeled_rewards = relabeled_rewards_before_reshape.reshape(
            #     num_tasks, self.num_network_ensemble, contexts.shape[0]
            # )


# April 7 2020
# sequential code to obtain the training target for critic, vae and perturbation
            # target_q = []
            # target_candidates = []
            # target_perturbations = []

            # torch.cuda.synchronize()

            # for i, batch_idx in enumerate(batch_idxes):

            #     tq = self.bcq_polices[batch_idx].critic.q1(
            #         obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], 
            #         actions[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size]
            #     ).detach()
            #     target_q.append(tq)

            #     tc = self.bcq_polices[batch_idx].vae.decode(
            #         obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], 
            #         z[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size]
            #     ).detach()
            #     target_candidates.append(tc)

            #     tp = self.bcq_polices[batch_idx].get_perturbation(
            #         obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], 
            #         tc
            #     ).detach()
            #     target_perturbations.append(tp)

            # target_q = torch.cat(target_q, dim=0).squeeze()
            # target_candidates = torch.cat(target_candidates, dim=0)
            # target_perturbations = torch.cat(target_perturbations, dim=0)

            # torch.cuda.synchronize()


# April 8 2020
# code to run data parallel to relabel transition
            # torch.cuda.synchronize()

            # # --------------------------------------------------------
            # data_parallel_rewards = torch.nn.parallel.data_parallel(
            #     self.ensemble_predictor, contexts_obs_actions
            # )

            # data_parallel_rewards = data_parallel_rewards.t().reshape(
            #     num_tasks, self.num_network_ensemble, contexts.shape[0]
            # )

            # torch.cuda.synchronize()


# April 16 2020

# codes to check the correctness of the reward prediction ensemble

# print('------------------------------------------')
# for i in range(num_tasks):
#     rewards_predicted = relabeled_rewards_mean[i, i*num_trans_context: (i+1)*num_trans_context]
#     gtruth = contexts[i*num_trans_context: (i+1)*num_trans_context, -1]
#     print(F.mse_loss(gtruth, rewards_predicted))

# print('------------------------------------------')

# exit()
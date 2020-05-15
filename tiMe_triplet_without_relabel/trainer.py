import gtimer as gt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict, deque

import utils.pytorch_util as ptu
from utils.pytorch_util import np_to_pytorch_batch


class SuperQTrainer(object):
    def __init__(
            self,
            nets,
            bcq_policies,
            num_trans_context,
            general_lr=3e-4,
            kl_lambda=.1,
            optimizer_class=optim.Adam,
    ):
        super().__init__()

        self.bcq_polices = bcq_policies
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

        self.kl_lambda = kl_lambda
        self.num_trans_context = num_trans_context
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train(self, batch, batch_idxes):

        """
        Unpack data from the batch
        """
        obs = batch['obs']
        actions = batch['actions']
        contexts = batch['contexts']

        num_candidate_context = contexts[0].shape[0]
        meta_batch_size = batch_idxes.shape[0]
        num_posterior = meta_batch_size * num_candidate_context

        # Get the in_mdp_batch_size
        in_mdp_batch_size = obs.shape[0] // batch_idxes.shape[0]

        # Sample z for each state
        z = self.bcq_polices[0].vae.sample_z(obs).to(ptu.device)

        target_q = []
        target_candidates = []
        target_perturbations = []

        for i, batch_idx in enumerate(batch_idxes):

            tq = self.bcq_polices[batch_idx].critic.q1(
                obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], 
                actions[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size]
            ).detach()
            target_q.append(tq)

            tc = self.bcq_polices[batch_idx].vae.decode(
                obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], 
                z[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size]
            ).detach()
            target_candidates.append(tc)

            tp = self.bcq_polices[batch_idx].get_perturbation(
                obs[i*in_mdp_batch_size: (i+1)*in_mdp_batch_size], tc
            ).detach()
            target_perturbations.append(tp)

        target_q = torch.cat(target_q, dim=0).squeeze()
        target_candidates = torch.cat(target_candidates, dim=0)
        target_perturbations = torch.cat(target_perturbations, dim=0)

        gt.stamp('get_the_targets', unique=False)

        """
        Compute triplet loss
        """
        self.context_encoder_optimizer.zero_grad()

        unscaled_triplet_loss = 0.
        posterior_mean_list = []
        posterior_var_list = []
        for c in contexts:

            # TODO: Dimension of z_means and z_vars
            z_means, z_vars = self.context_encoder.infer_posterior_with_mean_var(c)
            posterior_mean_list.append(z_means)
            posterior_var_list.append(z_vars)
        
        # TODO: expain dimension of posterior_mean_list and posterior_var_list

        All_kl = []
        for i in range(len(posterior_mean_list)):
            z_means = posterior_mean_list[i]
            z_vars = posterior_var_list[i]

            posterior_mean_list_copy = list(posterior_mean_list)
            posterior_mean_list_copy.pop(i)
            means_other_tasks = torch.cat(posterior_mean_list_copy, dim=0)

            posterior_var_list_copy = list(posterior_var_list)
            posterior_var_list_copy.pop(i)
            vars_other_tasks = torch.cat(posterior_var_list_copy, dim=0)

            for mean, var in zip(z_means, z_vars):
                result = []
                for m, v in zip(z_means, z_vars):
                    # if not torch.eq(mean, m) and not torch.eq(var, v):
                    kl = self.context_encoder.compute_kl_div_between_posterior(mean, var, m, v)
                    result.append(kl)
                    All_kl.append(kl)
                within_task_dist = torch.max(torch.stack(result))

                result = []
                for m, v in zip(means_other_tasks, vars_other_tasks):
                    kl = self.context_encoder.compute_kl_div_between_posterior(mean, var, m, v)
                    result.append(kl)
                    All_kl.append(kl)
                between_task_dist = torch.min(torch.stack(result))
                unscaled_triplet_loss += F.softplus(within_task_dist - between_task_dist)
            
        unscaled_triplet_loss /= num_posterior * 2

        
        # z_means_vec, z_var_vec: (num_posterior, latent_dim), num_posterior = meta_batch_size * num_candidate_context
        z_means_vec, z_vars_vec = self.context_encoder.infer_posterior_with_mean_var(torch.cat(contexts, dim=0))

        # z_means_interleave: (num_posterior * num_posterior, latent_dim) [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        z_means_interleave = torch.repeat_interleave(z_means_vec, num_posterior, dim=0)
        # z_means_repeat: (num_posterior * num_posterior, latent_dim) [1, 2, 3] -> [1, 2, 3, 2, 3, 1, 3, 1, 2].
        # By doing so, it is easy to get the triplet loss
        z_means_repeat = []
        for i in range(meta_batch_size):
            z_means_repeat.append(torch.cat([z_means_vec[i * num_candidate_context:], z_means_vec[:i * num_candidate_context]], dim=0).repeat(num_candidate_context, 1))
        z_means_repeat = torch.cat(z_means_repeat, dim=0)

        # As above
        z_vars_interleave = torch.repeat_interleave(z_vars_vec, num_posterior, dim=0)
        z_vars_repeat = []
        for i in range(meta_batch_size):
            z_vars_repeat.append(torch.cat([z_vars_vec[i * num_candidate_context:], z_vars_vec[:i * num_candidate_context]], dim=0).repeat(num_candidate_context, 1))
        z_vars_repeat = torch.cat(z_vars_repeat, dim=0)
        
        # log(det(Sigma2) / det(Sigma1)): (num_posterior * num_posterior, 1)
        kl_divergence = torch.log(torch.prod(z_vars_repeat / z_vars_interleave, dim=1))
        # -d
        kl_divergence -= z_means_vec.shape[-1]
        # Tr(Sigma2^{-1} * Sigma1)
        kl_divergence += torch.sum(z_vars_interleave / z_vars_repeat, dim=1)
        # (m2 - m1).T Sigma2^{-1} (m2 - m1))
        kl_divergence += torch.sum((z_means_repeat - z_means_interleave) ** 2 / z_vars_repeat, dim = 1)
        # / 2
        # (num_posterior, num_posterior): each element kl_{i, j} denotes the kl divergence between the two distributions.
        # Task number for row: i // num_posterior // num_candidate_context. 
        #             for col: j % num_posterior // num_candidate_context.
        # Batch number for row: i // num_posterior % num_candidate_context.
        #              for col: j % num_posterior % num_candidate_context.
        kl_divergence = kl_divergence.reshape(num_posterior, num_posterior) / 2

        unscaled_triplet_loss_vec = torch.sum(F.softplus(
            torch.max(kl_divergence[:, :num_candidate_context], dim=1)[0] - torch.min(kl_divergence[:, num_candidate_context:], dim=1)[0]
        ))
        
        # print(f'triplet_loss: {triplet_loss}, {triplet_loss_vec}')        
        # TODO: Explain the meaning of the coefficients
        
        gt.stamp('get_triplet_loss', unique=False)

        """
        Infer the context variables
        """
        inferred_mdps = []
        mean_var_list = []
        index = np.random.choice(num_candidate_context, meta_batch_size) + num_candidate_context * np.arange(meta_batch_size)
        i = 0
        for z_means, z_vars in zip(posterior_mean_list, posterior_var_list):
            idx = index[i]
            mean = z_means[idx - i * num_candidate_context]
            var = z_vars[idx - i * num_candidate_context]
            i += 1

            mean_var_list.append([mean, var])
            inferred_mdps.append(self.context_encoder.sample_z_from_mean_var(mean, var))
        
        inferred_mdps = torch.stack(inferred_mdps)
        inferred_mdps = torch.repeat_interleave(inferred_mdps, in_mdp_batch_size, dim=0)

        # Get the sampled mean and vars for each task.
        # mean: (meta_batch_size, latent_dim)
        # var: (meta_batch_size, latent_dim)
        mean = z_means_vec[index]
        var = z_vars_vec[index]

        # Get the inferred MDP
        # inferred_mdps_vec: (meta_batch_size, latent_dim)
        inferred_mdps_vec = self.context_encoder.sample_z_from_mean_var(mean, var)

        inferred_mdps_vec = torch.repeat_interleave(inferred_mdps_vec, in_mdp_batch_size, dim=0) 

        gt.stamp('infer_mdps', unique=False)
        """
        Obtain the KL loss
        """
        
        # KL constraint on z if probabilistic
        prior_mean = ptu.zeros(mean_var_list[0][0].shape)
        prior_var = ptu.ones(mean_var_list[0][1].shape)

        kl_div = []
        for m, v in mean_var_list:
            kl_div.append(self.context_encoder.compute_kl_div_between_posterior(m, v, prior_mean, prior_var))

        kl_div = torch.sum(torch.stack(kl_div))
        kl_loss = self.kl_lambda * kl_div

        prior_mean = ptu.zeros(mean.shape)
        prior_var = ptu.ones(var.shape)

        kl_loss_vec = self.kl_lambda * self.context_encoder.compute_kl_div_between_posterior(mean, var, prior_mean, prior_var)
        # for m, v in zip(mean, var):
        #     kl_div.append(self.context_encoder.compute_kl_div_between_posterior(m, v, prior_mean, prior_var))

        # kl_div = torch.sum(torch.stack(kl_div))
        # kl_loss_vec = self.kl_lambda * kl_div

        # print(f'kl_loss: {kl_loss}, {kl_loss_vec}')

        gt.stamp('get_kl_loss', unique=False)

        # triplet_loss = (kl_loss / unscaled_triplet_loss).detach() * unscaled_triplet_loss
        posterior_loss = unscaled_triplet_loss + kl_loss
        posterior_loss.backward(retain_graph=True)

        gt.stamp('get_kl_gradient', unique=False)
        """
        Obtain the Q-function loss
        """
        self.Qs_optimizer.zero_grad()
        pred_q = self.Qs(obs, actions, inferred_mdps)
        pred_q = torch.squeeze(pred_q)
        qf_loss = F.mse_loss(pred_q, target_q)

        gt.stamp('get_qf_loss', unique=False)

        qf_loss.backward()

        gt.stamp('get_qf_gradient', unique=False)

        self.Qs_optimizer.step()
        self.context_encoder_optimizer.step()
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

        candidate_loss = F.mse_loss(pred_candidates, target_candidates)
        perturbation_loss = F.mse_loss(pred_perturbations, target_perturbations)

        gt.stamp('get_candidate_and_perturbation_loss', unique=False)

        candidate_loss.backward()
        perturbation_loss.backward()

        gt.stamp('get_candidate_and_perturbation_gradient', unique=False)

        self.vae_decoder_optimizer.step()
        self.perturbation_generator_optimizer.step()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['unscaled_triplet_loss'] = np.mean(
                ptu.get_numpy(unscaled_triplet_loss)
            )
            self.eval_statistics['unscaled_triplet_loss_vec'] = np.mean(
                ptu.get_numpy(unscaled_triplet_loss_vec)
            )
            self.eval_statistics['kl_loss'] = np.mean(
                ptu.get_numpy(kl_loss)
            )
            self.eval_statistics['kl_loss_vec'] = np.mean(
                ptu.get_numpy(kl_loss_vec)
            )
            self.eval_statistics['qf_loss'] = np.mean(
                ptu.get_numpy(qf_loss)
            )
            self.eval_statistics['triplet_loss'] = np.mean(
                ptu.get_numpy(triplet_loss)
            )
            self.eval_statistics['kl_loss'] = np.mean(
                ptu.get_numpy(kl_loss)
            )
            self.eval_statistics['candidate_loss'] = np.mean(
                ptu.get_numpy(candidate_loss)
            )
            self.eval_statistics['perturbation_loss'] = np.mean(
                ptu.get_numpy(perturbation_loss)
            )
            
    def clear_z(self, num_tasks=1):
        self.context_encoder.clear_z(num_tasks)

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


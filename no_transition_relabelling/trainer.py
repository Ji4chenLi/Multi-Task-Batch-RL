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
            triplet_margin=2.0,
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
        self.triplet_margin = triplet_margin
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
        contexts = torch.cat(contexts, dim=0)

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

        # z_means, z_var: (num_posterior, latent_dim), num_posterior = meta_batch_size * num_candidate_context
        z_means, z_vars = self.context_encoder.infer_posterior_with_mean_var(contexts)

        # z_means_interleave: (num_posterior * num_posterior, latent_dim) [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        z_means_interleave = torch.repeat_interleave(z_means, num_posterior, dim=0)
        # z_means_repeat: (num_posterior * num_posterior, latent_dim) [1, 2, 3] -> [1, 2, 3, 2, 3, 1, 3, 1, 2].
        # By doing so, it is easy to get the triplet loss
        z_means_repeat = []
        for i in range(meta_batch_size):
            z_means_repeat.append(torch.cat([z_means[i * num_candidate_context:], z_means[:i * num_candidate_context]], dim=0).repeat(num_candidate_context, 1))
        z_means_repeat = torch.cat(z_means_repeat, dim=0)

        # As above
        z_vars_interleave = torch.repeat_interleave(z_vars, num_posterior, dim=0)
        z_vars_repeat = []
        for i in range(meta_batch_size):
            z_vars_repeat.append(torch.cat([z_vars[i * num_candidate_context:], z_vars[:i * num_candidate_context]], dim=0).repeat(num_candidate_context, 1))
        z_vars_repeat = torch.cat(z_vars_repeat, dim=0)

        gt.stamp('get_repeated_mean_var', unique=False)
        
        # log(det(Sigma2) / det(Sigma1)): (num_posterior * num_posterior, 1)
        kl_divergence = torch.log(torch.prod(z_vars_repeat / z_vars_interleave, dim=1))
        # -d
        kl_divergence -= z_means.shape[-1]
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

        within_task_dist = torch.max(kl_divergence[:, :num_candidate_context], dim=1)[0]
        across_task_dist = torch.min(kl_divergence[:, num_candidate_context:], dim=1)[0]

        unscaled_triplet_loss = torch.sum(F.relu(
            within_task_dist - across_task_dist + self.triplet_margin
        ))
        
        gt.stamp('get_triplet_loss', unique=False)

        """
        Infer the context variables
        """
        index = np.random.choice(num_candidate_context, meta_batch_size) + num_candidate_context * np.arange(meta_batch_size)
        # Get the sampled mean and vars for each task.
        # mean: (meta_batch_size, latent_dim)
        # var: (meta_batch_size, latent_dim)
        mean = z_means[index]
        var = z_vars[index]

        # Get the inferred MDP
        # inferred_mdps: (meta_batch_size, latent_dim)
        inferred_mdps = self.context_encoder.sample_z_from_mean_var(mean, var)

        inferred_mdps = torch.repeat_interleave(inferred_mdps, in_mdp_batch_size, dim=0) 

        gt.stamp('infer_mdps', unique=False)
        """
        Obtain the KL loss
        """
        prior_mean = ptu.zeros(mean.shape)
        prior_var = ptu.ones(var.shape)

        kl_loss = self.kl_lambda * self.context_encoder.compute_kl_div_between_posterior(mean, var, prior_mean, prior_var)

        gt.stamp('get_kl_loss', unique=False)

        # triplet_loss = (kl_loss / unscaled_triplet_loss).detach() * unscaled_triplet_loss
        # posterior_loss = unscaled_triplet_loss + kl_loss
        # posterior_loss.backward(retain_graph=True)

        # gt.stamp('get_posterior_gradient', unique=False)
        """
        Obtain the Q-function loss
        """
        self.Qs_optimizer.zero_grad()
        pred_q = self.Qs(obs, actions, inferred_mdps)
        pred_q = torch.squeeze(pred_q)
        qf_loss = F.mse_loss(pred_q, target_q)

        gt.stamp('get_qf_loss', unique=False)

        (qf_loss + unscaled_triplet_loss + kl_loss).backward()

        gt.stamp('get_qf_encoder_gradient', unique=False)

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
            self.eval_statistics['qf_loss'] = np.mean(
                ptu.get_numpy(qf_loss)
            )
            self.eval_statistics['unscaled_triplet_loss'] = np.mean(
                ptu.get_numpy(unscaled_triplet_loss)
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


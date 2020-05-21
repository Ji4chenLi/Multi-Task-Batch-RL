import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import utils.pytorch_util as ptu


inf_tensor = torch.tensor(np.inf)
inf_tensor = inf_tensor.to(torch.device('cuda:0'))


def assert_pt(a, b, msg):

    assert torch.all(torch.eq(a, b)) == 1, msg



def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''

    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _get_prod_of_gauss_mask(num_selected, desired_len):

    # Taken from
    # https://discuss.pytorch.org/t/create-a-2d-tensor-with-varying-lengths-of-one-in-each-row/25359

    # desired_length is the desired size of the second dimension of the masks

    seq_lens = ptu.from_numpy(
        np.array(num_selected)
    ).unsqueeze(-1)
    max_len = torch.max(seq_lens)

    # create tensor of suitable shape and same number of dimensions
    range_tensor = torch.arange(max_len).unsqueeze(0)
    range_tensor = range_tensor.to(ptu.device)
    range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))

    # until this step, we only created auxiliary tensors (you may already have from previous steps) 
    # the real mask tensor is created with binary masking:
    mask_tensor = (range_tensor < seq_lens) 

    mask_tensor = mask_tensor.type(torch.float)

    current_len = mask_tensor.shape[1]

    pad = ptu.zeros(mask_tensor.shape[0], desired_len - current_len)

    mask_tensor = torch.cat((mask_tensor, pad), dim=1)

    return mask_tensor


def _batch_product_of_gaussians(b_mus, b_vars, masks):

    # For first two input, each input has 3 dimensions
    # [batch size, number of transition in each batch, latent dim]

    # masks is 2 dimensional of size [batch size, number of transition]

    masks = masks.unsqueeze(-1)

    b_vars = torch.clamp(b_vars, min=1e-7)

    rec = torch.reciprocal(b_vars)

    rec_sum = torch.sum(
        rec * masks, dim=1
    )

    # set entries corresponding to 0.0 to 1.0
    # the loss term corresponding to 0.0 samples
    # should be excluded in the main training code
    rec_sum[rec_sum == 0.0] = 1.0

    one_over_rec_sum = 1.0 / rec_sum
    
    sum_mu_sigmas = torch.sum(
        (b_mus / b_vars) * masks, dim=1
    )    

    mu = one_over_rec_sum * sum_mu_sigmas

    return mu, one_over_rec_sum


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class ProbabilisticContextEncoder(nn.Module):

    def __init__(self,
                 mlp_encoder,
                 latent_dim,
    ):
        super().__init__()

        self.mlp_encoder = mlp_encoder
        self.latent_dim = latent_dim

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        var = ptu.ones(num_tasks, self.latent_dim)

        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_divs = torch.stack(kl_divs)
        return kl_divs

    def compute_kl_div_between_posterior(self, means_a, vars_a, means_b, vars_b):
        posteriors_a = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(means_a), torch.unbind(vars_a))]
        posteriors_b = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(means_b), torch.unbind(vars_b))]
        kl_divs = [torch.distributions.kl.kl_divergence(post_a, post_b) for post_a, post_b in zip(posteriors_a, posteriors_b)]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.mlp_encoder(context)
        params = params.view(context.size(0), -1, self.mlp_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)

        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        self.z_means = torch.stack([p[0] for p in z_params])
        self.z_vars = torch.stack([p[1] for p in z_params])

        self.sample_z()

    def infer_posterior_with_mean_var(self, context, num_trans_context, num_selected=None):
        params = self.mlp_encoder(context)
        params = params.view(context.size(0), -1, self.mlp_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)

        assert num_selected is not None

        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])

        # TODO: change 64 to num trans context
        batch_masks = _get_prod_of_gauss_mask(num_selected, num_trans_context)
        z_means, z_vars = _batch_product_of_gaussians(
            mu, sigma_squared, batch_masks
        )

        return z_means, z_vars

    def sample_z(self):
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        z = [d.rsample() for d in posteriors]
        self.z = torch.stack(z)

    def forward(self, context, return_prob_params=False):
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        if return_prob_params:
            return task_z, self.z_means, self.z_vars
            
        return task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return self.mlp_encoder

    def get_snapshot(self):
        return dict(
            mlp_encoder_state_dict=self.mlp_enconder.state_dict(),
        )


# April 9 2020
# Code to compute prod of gaussian sequentially
# to infer posterior
# using different number of transitions
# for each tasks

        #     z_params = []

        #     for idx, item in enumerate(zip(torch.unbind(mu), torch.unbind(sigma_squared))):
        #         m, s = item
        #         selected = num_selected[idx]
        #         pog = _product_of_gaussians(m[:selected], s[:selected])
        #         z_params.append(pog)

        # z_means = torch.stack([p[0] for p in z_params])
        # z_vars = torch.stack([p[1] for p in z_params])

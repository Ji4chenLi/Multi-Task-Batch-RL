"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

import utils.pytorch_util as ptu
from utils.core import PyTorchModule


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


# class MlpEncoder(PyTorchModule):
#     """
#     The input context should be a 3-D tensor.
#     Specifically,
#     1st dim: number of the contexts
#     2nd dim: number of transitions contained in context
#     3rd dim: ob + ac + reward
#     """
#     def __init__(
#         self,
#         g_hidden_sizes,
#         g_input_sizes,
#         h_hidden_sizes,
#         g_latent_dim,
#         latent_dim,
#     ):
#         super(MlpEncoder, self).__init__()
#         self.g_net = FlattenMlp(
#             hidden_sizes = g_hidden_sizes,
#             input_size=g_input_sizes,
#             output_size=g_latent_dim,
#         )
#         self.h_net = FlattenMlp(
#             hidden_sizes=h_hidden_sizes,
#             input_size=g_latent_dim,
#             output_size=latent_dim,
#         )
#         self.latent_dim=latent_dim

#     def forward(self, contexts):
#         z = self.g_net(contexts)
#         z = torch.mean(z, dim=1)
#         return self.h_net(z)


class PerturbationGenerator(FlattenMlp):
    """
    Generate the perturbation term for each
    input (s, a, inferred_mdp).
    """
    def __init__(
            self,
            max_action=1.0,
            perturbed_scale=0.05,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_activation = torch.tanh
        self.perturbed_scale = perturbed_scale
        self.max_action = max_action
    
    def forward(self, *inputs, **kwargs):
        output = super().forward(*inputs, **kwargs)
        return self.max_action * output

    def get_perturbed_actions(self, obs, actions, inferred_mdps):
        """
        Get actions + perturbations, which are further
        clamped by the max_action
        """
        perturbations = self.forward(obs, actions, inferred_mdps)
        return (actions + perturbations * self.perturbed_scale).clamp(-self.max_action, self.max_action)


class VaeDecoder(FlattenMlp):
    """
    Generate the candidate action for each
    input (s, inferred_mdp).
    """
    def __init__(
            self,
            max_action=1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_activation = torch.tanh
        self.max_action = max_action

    def forward(self, *inputs, **kwargs):
        output = super().forward(*inputs, **kwargs)
        return self.max_action * output

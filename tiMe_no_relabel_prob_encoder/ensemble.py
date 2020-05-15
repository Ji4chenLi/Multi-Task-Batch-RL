import torch
from torch import nn as nn
from torch.nn import functional as F
import utils.pytorch_util as ptu 
import numpy as np
from tqdm import tqdm

# def get_affine_params(ensemble_size, in_features, out_features):

#     w = truncated_normal(size=(ensemble_size, in_features, out_features),
#                          std=1.0 / (2.0 * np.sqrt(in_features)))
#     w = nn.Parameter(w)

#     b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

#     return w, b


# class EnsemblePredictor(nn.Module):

#     def __init__(self, ensemble_params_list, num_network_ensemble):
#         super().__init__()

#         self.lin0_w, self.lin0_b = get_affine_params(num_network_ensemble, in_features, 200)

#         self.lin1_w, self.lin1_b = get_affine_params(num_network_ensemble, 200, 200)

#         self.num_network_ensemble = num_network_ensemble

#     def forward(self, inputs):

#         # Transform inputs

#         inputs = inputs.matmul(self.lin0_w) + self.lin0_b
#         inputs = F.relu(inputs)

#         inputs = inputs.matmul(self.lin1_w) + self.lin1_b

#         return inputs

#     def _create_ensemble_list(self, ensemble_params_list):
#         self.lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)
#         self.lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
#         self.lin1_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)
#         self.lin1_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)


# class EnsemblePredictor(nn.Module):

#     def __init__(self, ensemble_params_list):
#         super().__init__()

#         self._create_ensemble_list(ensemble_params_list)

#     def forward_reshape(self, x):

#         # Transform inputs

#         inputs = x.matmul(self.lin0_w) + self.lin0_b
#         inputs = F.relu(inputs)

#         inputs = inputs.matmul(self.lin1_w) + self.lin1_b

#         return inputs.reshape(x.shape[0], -1)

#     def forward(self, x):
#         return self.forward_reshape(x)

#     def forward_no_reshape(self, inputs):

#         # Transform inputs

#         inputs = inputs.matmul(self.lin0_w) + self.lin0_b
#         inputs = F.relu(inputs)

#         inputs = inputs.matmul(self.lin1_w) + self.lin1_b

#         return inputs

#     def _create_ensemble_list(self, ensemble_params_list):
#         self.lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)
#         self.lin0_w = nn.Parameter(self.lin0_w)

#         self.lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
#         self.lin0_b = nn.Parameter(self.lin0_b)

#         self.lin1_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)
#         self.lin1_w = nn.Parameter(self.lin1_w)

#         self.lin1_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
#         self.lin1_b = nn.Parameter(self.lin1_b)


class EnsemblePredictor(nn.Module):

    def __init__(self, ensemble_params_list):
        super().__init__()

        self._create_ensemble_list(ensemble_params_list)

    def forward(self, inputs):

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = inputs.squeeze()
        inputs = inputs.t()

        return inputs

# ------------------------------------------------------
# Original forward function
    # def forward_original(self, inputs):

    #     inputs = inputs.matmul(self.lin0_w) + self.lin0_b
    #     inputs = F.relu(inputs)

    #     inputs = inputs.matmul(self.lin1_w) + self.lin1_b

    #     return inputs

# ------------------------------------------------------

    def forward_mul_device(self, inputs):

        chunked_inputs = torch.chunk(inputs, torch.cuda.device_count(), dim=0)
        results = []

        for i, c_i in enumerate(chunked_inputs):
            c_i = c_i.to(torch.device(f'cuda:{i}'))
          
            lin0_w, lin0_b, lin1_w, lin1_b = self.nets[i]
            r_i = c_i.matmul(lin0_w) + lin0_b
            r_i = F.relu(r_i)
            r_i = r_i.matmul(lin1_w) + lin1_b

            results.append(r_i)

        results = torch.cat(results, dim=1)
        return results

    def _create_ensemble_list(self, ensemble_params_list):

        nets = []

        for device in range(torch.cuda.device_count()):
            lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin1_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin1_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            params = [lin0_w, lin0_b, lin1_w, lin1_b]

            for i, p in enumerate(params):
                params[i] = p.to(torch.device(f'cuda:{device}'))

            nets.append(params)

        self.nets = nets

        self.lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)
        self.lin0_w = nn.Parameter(self.lin0_w)

        self.lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        self.lin0_b = nn.Parameter(self.lin0_b)

        self.lin1_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)
        self.lin1_w = nn.Parameter(self.lin1_w)

        self.lin1_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        self.lin1_b = nn.Parameter(self.lin1_b)


# April 7 2020
#  code to check correctness of batch matrix multiplication
    # def forward_sequential(self, inputs, expected_outputs):

    #     results = torch.empty(640, inputs.shape[0], 1)
    #     results = results.to(ptu.device)

    #     assert results.shape == expected_outputs.shape

    #     for sample_i, each_sample in enumerate(tqdm(inputs)):

    #         for net_i in range(self.lin0_w.shape[0]):

    #             lin0_w = self.lin0_w[net_i] 
    #             lin0_b = self.lin0_b[net_i]
    #             lin1_w = self.lin1_w[net_i]
    #             lin1_b = self.lin1_b[net_i]

    #             r = torch.matmul(each_sample, lin0_w) + lin0_b
    #             r = F.relu(r)

    #             r = torch.matmul(r, lin1_w) + lin1_b

    #             results[net_i, sample_i, 0] = r 

    #             expected_np = ptu.get_numpy(expected_outputs[net_i, sample_i, 0])
    #             r_np = ptu.get_numpy(r)

    #             assert np.allclose(expected_np, r_np, atol=0.001, rtol=0.001), (expected_outputs[net_i, sample_i, 0].item(), r.item())

    #     return results

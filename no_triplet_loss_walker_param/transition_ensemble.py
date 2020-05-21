import torch
from torch import nn as nn
from torch.nn import functional as F
import utils.pytorch_util as ptu 
import numpy as np
from tqdm import tqdm


class EnsembleTransitionPredictor(nn.Module):

    def __init__(self, ensemble_params_list):
        super().__init__()

        self._create_ensemble_list(ensemble_params_list)

    def forward(self, inputs):

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin4_w) + self.lin4_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin5_w) + self.lin5_b
        inputs = F.relu(inputs)

        inputs = inputs.matmul(self.lin6_w) + self.lin6_b

        # inputs = inputs.squeeze()
        # inputs = inputs.t()

        return inputs

    def forward_mul_device(self, inputs):

        chunked_inputs = torch.chunk(inputs, torch.cuda.device_count(), dim=0)
        results = []

        for i, c_i in enumerate(chunked_inputs):
            c_i = c_i.to(torch.device(f'cuda:{i}'))
          
            lin0_w, lin0_b, lin1_w, lin1_b, lin2_w, lin2_b, lin3_w, lin3_b, lin4_w, lin4_b, lin5_w, lin5_b, lin6_w, lin6_b = self.nets[i]

            r_i = c_i.matmul(lin0_w) + lin0_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin1_w) + lin1_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin2_w) + lin2_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin3_w) + lin3_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin4_w) + lin4_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin5_w) + lin5_b
            r_i = F.relu(r_i)

            r_i = r_i.matmul(lin6_w) + lin6_b


            results.append(r_i)

        results = torch.cat(results, dim=1)
        return results

    def _create_ensemble_list(self, ensemble_params_list):

        nets = []

        for device in range(torch.cuda.device_count()):
            lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin1_w = torch.cat([param['fc1.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin1_b = torch.cat([param['fc1.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin2_w = torch.cat([param['fc2.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin2_b = torch.cat([param['fc2.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin3_w = torch.cat([param['fc3.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin3_b = torch.cat([param['fc3.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin4_w = torch.cat([param['fc4.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin4_b = torch.cat([param['fc4.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin5_w = torch.cat([param['fc5.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin5_b = torch.cat([param['fc5.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            lin6_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)

            lin6_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)

            params = [lin0_w, lin0_b, lin1_w, lin1_b, lin2_w, lin2_b, lin3_w, lin3_b, lin4_w, lin4_b, lin5_w, lin5_b, lin6_w, lin6_b]

            for i, p in enumerate(params):
                params[i] = p.to(torch.device(f'cuda:{device}'))

            nets.append(params)

        self.nets = nets

        # self.lin0_w = torch.cat([param['fc0.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin0_w = nn.Parameter(self.lin0_w)

        # self.lin0_b = torch.cat([param['fc0.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin0_b = nn.Parameter(self.lin0_b)

        # self.lin1_w = torch.cat([param['fc1.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin1_w = nn.Parameter(self.lin1_w)

        # self.lin1_b = torch.cat([param['fc1.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin1_b = nn.Parameter(self.lin1_b)

        # self.lin2_w = torch.cat([param['fc2.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin2_w = nn.Parameter(self.lin2_w)

        # self.lin2_b = torch.cat([param['fc2.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin2_b = nn.Parameter(self.lin2_b)

        # self.lin3_w = torch.cat([param['fc3.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin3_w = nn.Parameter(self.lin3_w)

        # self.lin3_b = torch.cat([param['fc3.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin3_b = nn.Parameter(self.lin3_b)

        # self.lin4_w = torch.cat([param['fc4.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin4_w = nn.Parameter(self.lin4_w)

        # self.lin4_b = torch.cat([param['fc4.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin4_b = nn.Parameter(self.lin4_b)

        # self.lin5_w = torch.cat([param['fc5.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin5_w = nn.Parameter(self.lin5_w)

        # self.lin5_b = torch.cat([param['fc5.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin5_b = nn.Parameter(self.lin5_b)

        # self.lin6_w = torch.cat([param['last_fc.weight'].t()[None] for param in ensemble_params_list], dim=0)
        # self.lin6_w = nn.Parameter(self.lin6_w)

        # self.lin6_b = torch.cat([param['last_fc.bias'].reshape(1, -1)[None] for param in ensemble_params_list], dim=0)
        # self.lin6_b = nn.Parameter(self.lin6_b)

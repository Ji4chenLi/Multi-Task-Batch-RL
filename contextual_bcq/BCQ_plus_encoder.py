import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import gtimer as gt
from collections import OrderedDict
from utils.pytorch_util import get_numpy, zeros_like
from networks import FlattenMlp, MlpEncoder
from prob_context_encoder import ProbabilisticContextEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_latent_dim, hid_sizes, max_action):
		super(Actor, self).__init__()

		self.l_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

		self.linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

		self.l_out = nn.Linear(hid_sizes[-1], action_dim)
		
		self.max_action = max_action

	def forward(self, state, action, inferred_mdp):
		perturbation = self.get_perturbation(state, action, inferred_mdp)
		return (action + 0.05 * perturbation).clamp(-self.max_action, self.max_action)

	def get_perturbation(self, state, action, inferred_mdp):

		interm = F.relu(self.l_ipt(torch.cat([state, action, inferred_mdp], 1)))

		for net in self.linears:
			interm = F.relu(net(interm))

		interm = self.l_out(interm)
		interm = torch.tanh(interm) * self.max_action

		return interm


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, encoder_latent_dim,  hid_sizes):
        super(Critic, self).__init__()

        self.l1_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

        self.l1_linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

        self.l1_out = nn.Linear(hid_sizes[-1], 1)

        self.l2_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

        self.l2_linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

        self.l2_out = nn.Linear(hid_sizes[-1], 1)

    def forward(self, state, action, inferred_mdp):
        cat_input = torch.cat([state, action, inferred_mdp], 1)

        q1 = F.relu(self.l1_ipt(cat_input))

        for net in self.l1_linears:
            q1 = F.relu(net(q1))

        q1 = self.l1_out(q1)

        q2 = F.relu(self.l2_ipt(cat_input))

        for net in self.l2_linears:
            q2 = F.relu(net(q2))

        q2 = self.l2_out(q2)

        return q1, q2


    def q1(self, state, action, inferred_mdp):

        q1 = F.relu(self.l1_ipt(torch.cat([state, action, inferred_mdp], 1)))

        for net in self.l1_linears:
            q1 = F.relu(net(q1))

        q1 = self.l1_out(q1)

        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, encoder_latent_dim, vae_latent_dim, e_hid_sizes, d_hid_sizes, max_action):
        super(VAE, self).__init__()

        self.e_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, e_hid_sizes[0])

        self.e_linears = nn.ModuleList([nn.Linear(e_hid_sizes[i], e_hid_sizes[i+1]) for i in range(len(e_hid_sizes) - 1)])

        self.mean = nn.Linear(e_hid_sizes[-1], vae_latent_dim)
        self.log_std = nn.Linear(e_hid_sizes[-1], vae_latent_dim)

        self.d_ipt = nn.Linear(state_dim + vae_latent_dim + encoder_latent_dim, d_hid_sizes[0])

        self.d_linears = nn.ModuleList([nn.Linear(d_hid_sizes[i], d_hid_sizes[i+1]) for i in range(len(d_hid_sizes) - 1)])

        self.d_out = nn.Linear(d_hid_sizes[-1], action_dim)

        self.max_action = max_action
        self.vae_latent_dim = vae_latent_dim


    def forward(self, state, action, inferred_mdp):

        z = F.relu(self.e_ipt(torch.cat([state, action, inferred_mdp], 1)))

        for net in self.e_linears:
            z = F.relu(net(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        if torch.cuda.is_available():
            z = mean + std * torch.cuda.FloatTensor(size=(std.size())).normal_()
        else:
            print('Warning: using CPU!')
            z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 

        u = self.decode(state, z, inferred_mdp)

        return u, mean, std


    def decode(self, state, z=None, inferred_mdp=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = self.sample_z(state)

        a = F.relu(self.d_ipt(torch.cat([state, z, inferred_mdp], 1)))

        for net in self.d_linears:
            a = F.relu(net(a))

        a = self.d_out(a)

        return self.max_action * torch.tanh(a)
	
	
    def sample_z(self, state):
        if torch.cuda.is_available():
            z = torch.cuda.FloatTensor(state.size(0), self.vae_latent_dim).normal_().clamp(-0.5, 0.5)
        else:
            print('Warning: using CPU!')
            z = torch.randn(size=(state.size(0), self.vae_latent_dim)).to(device).clamp(-0.5, 0.5)

        return z


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, vae_latent_dim_multiplicity, target_q_coef,
                    actor_hid_sizes, critic_hid_sizes, vae_e_hid_sizes, vae_d_hid_sizes, encoder_latent_dim, 
        ):

        vae_latent_dim = vae_latent_dim_multiplicity * action_dim
        self.actor = Actor(state_dim, action_dim, encoder_latent_dim, actor_hid_sizes, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, encoder_latent_dim, actor_hid_sizes, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, encoder_latent_dim, critic_hid_sizes).to(device)
        self.critic_target = Critic(state_dim, action_dim, encoder_latent_dim, critic_hid_sizes).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.vae = VAE(state_dim, action_dim, encoder_latent_dim, vae_latent_dim,
                        vae_e_hid_sizes, vae_d_hid_sizes, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4) 

        mlp_enconder_input_size = 2 * state_dim + action_dim + 1

        mlp_enconder = MlpEncoder(
            hidden_sizes=[200, 200, 200],
            input_size=mlp_enconder_input_size,
            output_size=2 * encoder_latent_dim
        )
        self.context_encoder = ProbabilisticContextEncoder(
            mlp_enconder,
            encoder_latent_dim
        )
        self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=3e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.target_q_coef = target_q_coef

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def get_perturbation(self, state, action, inferred_mdp):
        perturbation = self.actor.get_perturbation(state, action, inferred_mdp)
        return perturbation

    def select_action(self, state, inferred_mdp):		
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            inferred_mdp = torch.FloatTensor(inferred_mdp.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state, self.vae.decode(state, inferred_mdp=inferred_mdp), inferred_mdp)
            q1 = self.critic.q1(state, action, inferred_mdp)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def train(self, train_data, discount=0.99, tau=0.005):
        state_np, next_state_np, action, reward, done, context = train_data
        state 		= torch.FloatTensor(state_np).to(device)
        action 		= torch.FloatTensor(action).to(device)
        next_state 	= torch.FloatTensor(next_state_np).to(device)
        reward 		= torch.FloatTensor(reward).to(device)
        done 		= torch.FloatTensor(1 - done).to(device)
        context     = torch.FloatTensor(context).to(device)

        gt.stamp('unpack_data', unique=False)

        # Infer mdep identity using context

        self.context_encoder_optimizer.zero_grad()

        inferred_mdp = self.context_encoder(context)
        in_mdp_batch_size = state.shape[0] // context.shape[0]
        inferred_mdp = torch.repeat_interleave(inferred_mdp, in_mdp_batch_size, dim=0)

        gt.stamp('infer_mdp_identity', unique=False)

        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action, inferred_mdp)
        recon_loss = F.mse_loss(recon, action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        gt.stamp('get_vae_loss', unique=False)

        self.vae_optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        self.vae_optimizer.step()

        gt.stamp('update_vae', unique=False)

        # Critic Training
        self.critic_optimizer.zero_grad()

        with torch.no_grad():

            # Duplicate state 10 times
            state_rep = next_state.repeat_interleave(10, dim=0)
            inferred_mdp_rep = inferred_mdp.repeat_interleave(10, dim=0)

            target_Q1, target_Q2 = self.critic_target(
                state_rep, self.actor_target(state_rep, self.vae.decode(state_rep, inferred_mdp=inferred_mdp_rep), inferred_mdp_rep), inferred_mdp_rep
            )

            # Soft Clipped Double Q-learning 
            target_Q = self.target_q_coef * torch.min(target_Q1, target_Q2) + (1 - self.target_q_coef) * torch.max(target_Q1, target_Q2)
            target_Q = target_Q.view(state.shape[0], -1).max(1)[0].view(-1, 1)

            target_Q = reward + done * discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action, inferred_mdp)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        gt.stamp('get_critic_loss', unique=False)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        gt.stamp('update_critic', unique=False)

        self.context_encoder_optimizer.step()

        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state, inferred_mdp=inferred_mdp.detach())
        perturbed_actions = self.actor(state, sampled_actions, inferred_mdp.detach())

        # Update through DPG
        actor_loss = -self.critic.q1(state, perturbed_actions, inferred_mdp.detach()).mean()

        gt.stamp('get_actor_loss', unique=False)
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        gt.stamp('update_actor', unique=False)
        
        # Update Target Networks 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        """
        Save some statistics for eval   
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['actor_loss'] = np.mean(
                get_numpy(actor_loss)
            )
            self.eval_statistics['critic_loss'] = np.mean(
                get_numpy(critic_loss)
            )
            self.eval_statistics['vae_loss'] = np.mean(
                get_numpy(vae_loss)
            )
        
    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor, self.critic, self.vae, self.context_encoder.mlp_encoder]

    def get_diagnostics(self):
        return self.eval_statistics

    def get_snapshot(self):
        return dict(
            actor_dict=self.actor.state_dict(),
            critic_dict=self.critic.state_dict(),
            vae_dict=self.vae.state_dict(),
            context_encoder_dict=self.context_encoder.state_dict(),
            eval_statistics=self.eval_statistics,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
        )

    def restore_from_snapshot(self, ss):
        ss = ss['trainer']
        
        self.actor.load_state_dict(ss['actor_dict'])
        self.actor.to(device)

        self.critic.load_state_dict(ss['critic_dict'])
        self.critic.to(device)

        self.vae.load_state_dict(ss['vae_dict'])
        self.vae.to(device)

        self.context_encoder.load_state_dict(ss['context_encoder_dict'])
        self.context_encoder.to(device)

        self.eval_statistics = ss['eval_statistics']
        self._need_to_update_eval_statistics = ss['_need_to_update_eval_statistics']


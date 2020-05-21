import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from collections import OrderedDict
import gtimer as gt

from utils.pytorch_util import get_numpy, zeros_like
from networks import FlattenMlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_latent_dim, hid_sizes, max_action):
		super(Actor, self).__init__()

		self.l_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

		self.linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

		self.l_out = nn.Linear(hid_sizes[-1], action_dim)

		# self.l1 = nn.Linear(state_dim + action_dim, 400)
		# self.l2 = nn.Linear(400, 300)
		# self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	def forward(self, state, action):
		perturbation = self.get_perturbation(state, action)
		return (action + 0.05 * perturbation).clamp(-self.max_action, self.max_action)

	def get_perturbation(self, state, action):
		# a = F.relu(self.l1(torch.cat([state, action], 1)))
		# a = F.relu(self.l2(a))
		# a = self.max_action * torch.tanh(self.l3(a))

		interm = F.relu(self.l_ipt(torch.cat([state, action], 1)))

		for net in self.linears:
			interm = F.relu(net(interm))

		interm = self.l_out(interm)
		interm = torch.tanh(interm) * self.max_action

		return interm


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_latent_dim, hid_sizes):
		super(Critic, self).__init__()
		# self.l1 = nn.Linear(state_dim + action_dim, 400)
		# self.l2 = nn.Linear(400, 300)
		# self.l3 = nn.Linear(300, 1)

		self.l1_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

		self.l1_linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

		self.l1_out = nn.Linear(hid_sizes[-1], 1)

		# self.l4 = nn.Linear(state_dim + action_dim, 400)
		# self.l5 = nn.Linear(400, 300)
		# self.l6 = nn.Linear(300, 1)

		self.l2_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, hid_sizes[0])

		self.l2_linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

		self.l2_out = nn.Linear(hid_sizes[-1], 1)

	def forward(self, state, action):
		# q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		# q1 = F.relu(self.l2(q1))
		# q1 = self.l3(q1)

		# q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		# q2 = F.relu(self.l5(q2))
		# q2 = self.l6(q2)

		q1 = F.relu(self.l1_ipt(torch.cat([state, action], 1)))

		for net in self.l1_linears:
			q1 = F.relu(net(q1))

		q1 = self.l1_out(q1)

		q2 = F.relu(self.l2_ipt(torch.cat([state, action], 1)))

		for net in self.l2_linears:
			q2 = F.relu(net(q2))

		q2 = self.l2_out(q2)

		return q1, q2


	def q1(self, state, action):
		# q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		# q1 = F.relu(self.l2(q1))
		# q1 = self.l3(q1)
		q1 = F.relu(self.l1_ipt(torch.cat([state, action], 1)))

		for net in self.l1_linears:
			q1 = F.relu(net(q1))

		q1 = self.l1_out(q1)

		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_latent_dim, vae_latent_dim, e_hid_sizes, d_hid_sizes, max_action):
		super(VAE, self).__init__()
		# self.e1 = nn.Linear(state_dim + action_dim, 750)
		# self.e2 = nn.Linear(750, 750)

		self.e_ipt = nn.Linear(state_dim + action_dim + encoder_latent_dim, e_hid_sizes[0])

		self.e_linears = nn.ModuleList([nn.Linear(e_hid_sizes[i], e_hid_sizes[i+1]) for i in range(len(e_hid_sizes) - 1)])

		self.mean = nn.Linear(e_hid_sizes[-1], vae_latent_dim)
		self.log_std = nn.Linear(e_hid_sizes[-1], vae_latent_dim)

		# self.mean = nn.Linear(750, vae_latent_dim)
		# self.log_std = nn.Linear(750, vae_latent_dim)

		# self.d1 = nn.Linear(state_dim + vae_latent_dim, 750)
		# self.d2 = nn.Linear(750, 750)
		# self.d3 = nn.Linear(750, action_dim)

		self.d_ipt = nn.Linear(state_dim + vae_latent_dim + encoder_latent_dim, d_hid_sizes[0])

		self.d_linears = nn.ModuleList([nn.Linear(d_hid_sizes[i], d_hid_sizes[i+1]) for i in range(len(d_hid_sizes) - 1)])

		self.d_out = nn.Linear(d_hid_sizes[-1], action_dim)

		self.max_action = max_action
		self.vae_latent_dim = vae_latent_dim


	def forward(self, state, action):
		# z = F.relu(self.e1(torch.cat([state, action], 1)))
		# z = F.relu(self.e2(z))

		z = F.relu(self.e_ipt(torch.cat([state, action], 1)))

		for net in self.e_linears:
			z = F.relu(net(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		if torch.cuda.is_available():
			z = mean + std * torch.cuda.FloatTensor(size=(std.size())).normal_()
		else:
			print(f'Using {device}')
			z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = self.sample_z(state)
			
		# a = F.relu(self.d1(torch.cat([state, z], 1)))
		# a = F.relu(self.d2(a))
		# return self.max_action * torch.tanh(self.d3(a))

		a = F.relu(self.d_ipt(torch.cat([state, z], 1)))

		for net in self.d_linears:
			a = F.relu(net(a))

		a = self.d_out(a)

		return self.max_action * torch.tanh(a)
	
	
	def sample_z(self, state):
		if torch.cuda.is_available():
			z = torch.cuda.FloatTensor(state.size(0), self.vae_latent_dim).normal_().clamp(-0.5, 0.5)
		else:
			z = torch.randn(size=(state.size(0), self.vae_latent_dim)).to(device).clamp(-0.5, 0.5)
		# z = torch.FloatTensor(
		# 	np.random.normal(0, 1, size=(state.size(0), self.vae_latent_dim))
		# ).to(device).clamp(-0.5, 0.5)
		return z

class MlpEncoder(nn.Module):
	def __init__(
		self,
		state_dim, 
		action_dim, 
		encoder_latent_dim,
        g_hid_sizes,
		g_latent_dim,
        h_hid_sizes,
	):
		super(MlpEncoder, self).__init__()
		self.g_ipt = nn.Linear(state_dim + action_dim + 1, g_hid_sizes[0])

		self.g_linears = nn.ModuleList([nn.Linear(g_hid_sizes[i], g_hid_sizes[i+1]) for i in range(len(g_hid_sizes) - 1)])

		self.g_out = nn.Linear(g_hid_sizes[-1], g_latent_dim)

		self.h_ipt = nn.Linear(g_latent_dim, h_hid_sizes[0])

		self.h_linears = nn.ModuleList([nn.Linear(h_hid_sizes[i], h_hid_sizes[i+1]) for i in range(len(h_hid_sizes) - 1)])

		self.h_out = nn.Linear(h_hid_sizes[-1], encoder_latent_dim)

		self.encoder_latent_dim = encoder_latent_dim

	def forward(self, contexts):
		z = F.relu(self.g_ipt(contexts))
		for net in self.g_linears:
			z = F.relu(net(z))
		z = self.g_out(z)

		z = torch.mean(z, dim=1)

		z = F.relu(self.h_ipt(z))
		for net in self.h_linears:
			z = F.relu(net(z))
		z = self.h_out(z)

		return z


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, vae_latent_dim_multiplicity, target_q_coef,
				 actor_hid_sizes, critic_hid_sizes, vae_e_hid_sizes, vae_d_hid_sizes, encoder_latent_dim, 
				 g_hid_sizes, g_latent_dim, h_hid_sizes, E_hid_sizes, P_hid_sizes):

		vae_latent_dim = vae_latent_dim_multiplicity * action_dim
		self.actor = Actor(state_dim, action_dim, encoder_latent_dim, actor_hid_sizes, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, encoder_latent_dim, actor_hid_sizes, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.critic = Critic(state_dim, action_dim, encoder_latent_dim, critic_hid_sizes).to(device)
		self.critic_target = Critic(state_dim, action_dim, encoder_latent_dim, critic_hid_sizes).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.vae = VAE(state_dim, action_dim, encoder_latent_dim, vae_latent_dim,
					   vae_e_hid_sizes, vae_d_hid_sizes, max_action).to(device)

		self.mlp_encoder = MlpEncoder(
			state_dim, action_dim, encoder_latent_dim, g_hid_sizes, g_latent_dim, h_hid_sizes
		).to(device)

		self.E = FlattenMlp(
			hidden_sizes=E_hid_sizes,
			input_size=state_dim + action_dim,
			output_size=state_dim,
		)
		self.P = FlattenMlp(
			hidden_sizes=P_hid_sizes,
			input_size=state_dim + encoder_latent_dim,
			output_size=1,
		)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4)
		self.mlp_encoder_optimizer = torch.optim.Adam(self.mlp_encoder.parameters(), lr=3e-4)
		self.E_optimizer = torch.optim.Adam(self.E.parameters(), lr=3e-4)
		self.P_optimizer = torch.optim.Adam(self.P.parameters(), lr=3e-4)

		self.max_action = max_action
		self.action_dim = action_dim
		self.target_q_coef = target_q_coef

		self._need_to_update_eval_statistics = True
		self.eval_statistics = OrderedDict()

	def get_perturbation(self, state, action):
		perturbation = self.actor.get_perturbation(state, action)
		return perturbation

	def select_action(self, state):	
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.max(0)[1]
		return action[ind].cpu().data.numpy().flatten()

	def train(self, train_data, discount=0.99, tau=0.005):

		# Sample replay buffer / batch
		state_np, next_state_np, action, reward, done, context = train_data
		state 		= torch.FloatTensor(state_np).to(device)
		action 		= torch.FloatTensor(action).to(device)
		next_state 	= torch.FloatTensor(next_state_np).to(device)
		reward 		= torch.FloatTensor(reward).to(device)
		done 		= torch.FloatTensor(1 - done).to(device)
		context     = torch.FloatTensor(context).to(device)

		gt.stamp('unpack_data', unique=False)

		# Infer mdep identity using context
		# inferred_mdp = self.mlp_encoder(context)
		# in_mdp_batch_size = state.shape[0] // context.shape[0]
		# inferred_mdp = torch.repeat_interleave(inferred_mdp, in_mdp_batch_size, dim=0)

		# gt.stamp('infer_mdp_identity', unique=False)

		# Train the mlp encoder to predict the rewards.
		# self.mlp_encoder.zero_grad()
		# pred_next_obs = self.E(state, action)
		# pred_rewards = self.P(pred_next_obs, inferred_mdp)
		# reward_loss = F.mse_loss(pred_rewards, reward)

		# gt.stamp('get_reward_loss', unique=False)

		# reward_loss.backward(retain_graph=True)

		# gt.stamp('get_reward_gradient', unique=False)

		# Extend the state space using the inferred_mdp
		# state = torch.cat([state, inferred_mdp], dim=1)
		# next_state = torch.cat([next_state, inferred_mdp], dim=1)

		# gt.stamp('extend_original_state', unique=False)

		# Critic Training
		self.critic_optimizer.zero_grad()
		with torch.no_grad():

			# Duplicate state 10 times
			state_rep = next_state.repeat_interleave(10, dim=0)
			gt.stamp('check0', unique=False)

			# candidate_action = self.vae.decode(state_rep)
			# torch.cuda.synchronize()
			# gt.stamp('check1', unique=False)

			# perturbated_action = self.actor_target(state_rep, candidate_action)
			# torch.cuda.synchronize()
			# gt.stamp('check2', unique=False)

			# target_Q1, target_Q2 = self.critic_target(state_rep, perturbated_action)
			# torch.cuda.synchronize()
			# gt.stamp('check3', unique=False)

			target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))
			
			# Soft Clipped Double Q-learning 
			target_Q = self.target_q_coef * torch.min(target_Q1, target_Q2) + (1 - self.target_q_coef) * torch.max(target_Q1, target_Q2)

			target_Q = target_Q.view(state.shape[0], -1).max(1)[0].view(-1, 1)

			target_Q = reward + done * discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		gt.stamp('get_critic_loss', unique=False)

		critic_loss.backward() # retain_graph=True
		gt.stamp('get_critic_gradient', unique=False)

		# self.mlp_encoder_optimizer.step()
		# gt.stamp('update_mlp_encoder', unique=False)
		
		# Variational Auto-Encoder Training
		recon, mean, std = self.vae(state, action)
		recon_loss = F.mse_loss(recon, action)
		KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
		vae_loss = recon_loss + 0.5 * KL_loss

		gt.stamp('get_vae_loss', unique=False)

		self.vae_optimizer.zero_grad()
		vae_loss.backward()
		self.vae_optimizer.step()

		gt.stamp('update_vae', unique=False)

		self.critic_optimizer.step()
		gt.stamp('update_critic', unique=False)

		# Pertubation Model / Action Training
		sampled_actions = self.vae.decode(state)
		perturbed_actions = self.actor(state, sampled_actions)

		# Update through DPG
		self.actor_optimizer.zero_grad()
		actor_loss = -self.critic.q1(state, perturbed_actions).mean()

		gt.stamp('get_actor_loss', unique=False)

		self.actor_optimizer.step()
		gt.stamp('update_actor', unique=False)

		# Update Target Networks 
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		gt.stamp('update_target_network', unique=False)

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
			# self.eval_statistics['reward_loss'] = np.mean(
			# 	get_numpy(reward_loss)
			# )
			
	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		self._need_to_update_eval_statistics = True
	
	@property
	def networks(self):
		return [self.actor, self.critic, self.vae, self.mlp_encoder, self.E, self.P]

	@property
	def eval_networks(self):
		'''
		Return networks for the policy evaluation
		'''
		return [self.actor, self.critic, self.vae, self.mlp_encoder]

	def get_snapshot(self):
		return dict(
			actor_dict=self.actor.state_dict(),
			critic_dict=self.critic.state_dict(),
			vae_dict=self.vae.state_dict(),
			mlp_encoder_dict=self.mlp_encoder.state_dict(),
			E_state_dict=self.E.state_dict(),
			P_state_dict=self.P.state_dict(),
			joint_optimizer_state_dict=self.joint_optimizer.state_dict(),

			eval_statistics=self.eval_statistics,
			_need_to_update_eval_statistics=self._need_to_update_eval_statistics
		)
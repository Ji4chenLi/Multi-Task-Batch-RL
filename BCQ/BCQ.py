import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_sizes, max_action):
		super(Actor, self).__init__()

		self.l_ipt = nn.Linear(state_dim + action_dim, hid_sizes[0])

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
	def __init__(self, state_dim, action_dim, hid_sizes):
		super(Critic, self).__init__()
		# self.l1 = nn.Linear(state_dim + action_dim, 400)
		# self.l2 = nn.Linear(400, 300)
		# self.l3 = nn.Linear(300, 1)

		self.l1_ipt = nn.Linear(state_dim + action_dim, hid_sizes[0])

		self.l1_linears = nn.ModuleList([nn.Linear(hid_sizes[i], hid_sizes[i+1]) for i in range(len(hid_sizes) - 1)])

		self.l1_out = nn.Linear(hid_sizes[-1], 1)

		# self.l4 = nn.Linear(state_dim + action_dim, 400)
		# self.l5 = nn.Linear(400, 300)
		# self.l6 = nn.Linear(300, 1)

		self.l2_ipt = nn.Linear(state_dim + action_dim, hid_sizes[0])

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
	def __init__(self, state_dim, action_dim, latent_dim, e_hid_sizes, d_hid_sizes, max_action):
		super(VAE, self).__init__()
		# self.e1 = nn.Linear(state_dim + action_dim, 750)
		# self.e2 = nn.Linear(750, 750)

		self.e_ipt = nn.Linear(state_dim + action_dim, e_hid_sizes[0])

		self.e_linears = nn.ModuleList([nn.Linear(e_hid_sizes[i], e_hid_sizes[i+1]) for i in range(len(e_hid_sizes) - 1)])

		self.mean = nn.Linear(e_hid_sizes[-1], latent_dim)
		self.log_std = nn.Linear(e_hid_sizes[-1], latent_dim)

		# self.mean = nn.Linear(750, latent_dim)
		# self.log_std = nn.Linear(750, latent_dim)

		# self.d1 = nn.Linear(state_dim + latent_dim, 750)
		# self.d2 = nn.Linear(750, 750)
		# self.d3 = nn.Linear(750, action_dim)

		self.d_ipt = nn.Linear(state_dim + latent_dim, d_hid_sizes[0])

		self.d_linears = nn.ModuleList([nn.Linear(d_hid_sizes[i], d_hid_sizes[i+1]) for i in range(len(d_hid_sizes) - 1)])

		self.d_out = nn.Linear(d_hid_sizes[-1], action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim


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
			z = torch.cuda.FloatTensor(state.size(0), self.latent_dim).normal_().clamp(-0.5, 0.5)
		else:
			z = torch.randn(size=(state.size(0), self.latent_dim)).to(device).clamp(-0.5, 0.5)
		# z = torch.FloatTensor(
		# 	np.random.normal(0, 1, size=(state.size(0), self.latent_dim))
		# ).to(device).clamp(-0.5, 0.5)
		return z


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, latent_dim_multiplicity, target_q_coef,
				 actor_hid_sizes, critic_hid_sizes, vae_e_hid_sizes, vae_d_hid_sizes):

		latent_dim = latent_dim_multiplicity * action_dim
		self.actor = Actor(state_dim, action_dim, actor_hid_sizes, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, actor_hid_sizes, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, critic_hid_sizes).to(device)
		self.critic_target = Critic(state_dim, action_dim, critic_hid_sizes).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.vae = VAE(state_dim, action_dim, latent_dim,
					   vae_e_hid_sizes, vae_d_hid_sizes, max_action).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.target_q_coef = target_q_coef

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

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		for it in trange(iterations):

			# Sample replay buffer / batch
			state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
			# state 		= torch.FloatTensor(state_np).to(device)
			# action 		= torch.FloatTensor(action).to(device)
			# next_state 	= torch.FloatTensor(next_state_np).to(device)
			# reward 		= torch.FloatTensor(reward).to(device)
			# done 		= torch.FloatTensor(1 - done).to(device)

			state 		= torch.from_numpy(state_np).float().to(device)
			action 		= torch.from_numpy(action).float().to(device)
			next_state 	= torch.from_numpy(next_state_np).float().to(device)
			reward 		= torch.from_numpy(reward).float().to(device)
			done 		= torch.from_numpy(1 - done).float().to(device)


			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():

				# Duplicate state 10 times
				# state_rep_before = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)

				# next_state_np_cpu = torch.FloatTensor(next_state_np)
				# # torch.cuda.synchronize()

				# next_state_np_gpu = next_state_np_cpu.to(device)
				# torch.cuda.synchronize()

				state_rep = next_state.repeat_interleave(10, dim=0)
				# torch.cuda.synchronize()

				# state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
				# print(state_rep.size())
				# print(torch.all(torch.eq(state_rep_before, state_rep)))
				# exit()
				# Compute value of perturbed actions sampled from the VAE
				# candidate_actions = self.vae.decode(state_rep)
				# perturbated_actions = self.actor_target(state_rep, candidate_actions)
				target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))

				# Soft Clipped Double Q-learning 
				target_Q = self.target_q_coef * torch.min(target_Q1, target_Q2) + (1 - self.target_q_coef) * torch.max(target_Q1, target_Q2)
				target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

				target_Q = reward + done * discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

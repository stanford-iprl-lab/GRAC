import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, device, noise_percent=0.):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

		self.device=device
		self.noise_percent = noise_percent
		self.noise_flag = False
		print('noise percent', self.noise_percent)

	def add_noise(self, x):
		if self.noise_flag:
			noise = self.noise_percent * x * torch.randn(x.shape, device=self.device)
			return x + noise
		else:
			return x

	def forward(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = self.add_noise(q1)
		q1 = F.relu(self.l2(q1))
		q1 = self.add_noise(q1)
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = self.add_noise(q2)
		q2 = F.relu(self.l5(q2))
		q2 = self.add_noise(q2)
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = self.add_noise(q1)
		q1 = F.relu(self.l2(q1))
		q1 = self.add_noise(q1)
		q1 = self.l3(q1)
		return q1


class GRAC(object):
	def __init__(
		self,
		env,
		state_dim,
		action_dim,
		max_action,
		batch_size=256,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		n_repeat=1,
		no_critic_cem=False,
		device='cuda:0',
		model_noise=0,
		alpha_start=0,
		alpha_end=0,
	):

		self.model_noise = model_noise
		print('model noise', self.model_noise)
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, device=device, noise_percent=model_noise).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		self.device = device
		self.log_freq = 200
		self.loss_rep = n_repeat

		self.loss_rep = 1
		self.policy_freq = 2
		
		print('loss rep', self.loss_rep)
		
	def select_action(self, state, writer=None, test=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100, writer=None, reward_range=20.0):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		self.critic.noise_flag = False
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q_final = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		for _ in range(self.loss_rep):
			current_Q1, current_Q2 = self.critic(state, action)
			
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
	
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			self.critic.noise_flag = True

			# Compute actor loss
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if log_it:
			with torch.no_grad():
				writer.add_scalar('train_critic/critic_loss', critic_loss, self.total_it)
	
				target_current_Q1_diff = target_Q1 - current_Q1 
				writer.add_scalar('q_diff/target_current_Q1_diff_max', target_current_Q1_diff.max(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q1_diff_min', target_current_Q1_diff.min(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q1_diff_mean', target_current_Q1_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q1_diff_abs_mean', target_current_Q1_diff.abs().mean(), self.total_it)
	
				target_current_Q2_diff = target_Q2 - current_Q2 
				writer.add_scalar('q_diff/target_current_Q2_diff_max', target_current_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q2_diff_min', target_current_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q2_diff_mean', target_current_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/target_current_Q2_diff_abs_mean', target_current_Q2_diff.abs().mean(), self.total_it)
	
				target_Q1_Q2_diff = target_Q1 - target_Q2
				writer.add_scalar('q_diff/target_Q1_Q2_diff_max', target_Q1_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_min', target_Q1_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_mean', target_Q1_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_abs_mean', target_Q1_Q2_diff.abs().mean(), self.total_it)
	
				current_Q1_Q2_diff = current_Q1 - current_Q2
				writer.add_scalar('q_diff/current_Q1_Q2_diff_max', current_Q1_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_min', current_Q1_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_mean', current_Q1_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_abs_mean', current_Q1_Q2_diff.abs().mean(), self.total_it)
	
				loss1_diff = target_Q_final - current_Q1
				writer.add_scalar('losses/loss1_diff_max', loss1_diff.max(), self.total_it)
				writer.add_scalar('losses/loss1_diff_min', loss1_diff.min(), self.total_it)
				writer.add_scalar('losses/loss1_diff_mean', loss1_diff.mean(), self.total_it)
				writer.add_scalar('losses/loss1_diff_abs_mean', loss1_diff.abs().mean(), self.total_it)
				
				loss2_diff = target_Q_final - current_Q2
				writer.add_scalar('losses/loss2_diff_max', loss2_diff.max(), self.total_it)
				writer.add_scalar('losses/loss2_diff_min', loss2_diff.min(), self.total_it)
				writer.add_scalar('losses/loss2_diff_mean', loss2_diff.mean(), self.total_it)
				writer.add_scalar('losses/loss2_diff_abs_mean', loss2_diff.abs().mean(), self.total_it)
	
				#target_Q
				writer.add_scalar('train_critic/target_Q/mean', torch.mean(target_Q), self.total_it)
				writer.add_scalar('train_critic/target_Q/max', target_Q.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q/min', target_Q.min(), self.total_it)
				writer.add_scalar('train_critic/target_Q/std', torch.std(target_Q), self.total_it)
	
				#target_Q1
				writer.add_scalar('train_critic/target_Q1/mean', torch.mean(target_Q1), self.total_it)
				writer.add_scalar('train_critic/target_Q1/max', target_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/min', target_Q1.min(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/std', torch.std(target_Q1), self.total_it)
	
				#target_Q2
				writer.add_scalar('train_critic/target_Q2/mean', torch.mean(target_Q2), self.total_it)
	
				#current_Q1
				writer.add_scalar('train_critic/current_Q1/mean', current_Q1.mean(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/std', torch.std(current_Q1), self.total_it)
				writer.add_scalar('train_critic/current_Q1/max', current_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/min', current_Q1.min(), self.total_it)
	
				# current_Q2
				writer.add_scalar('train_critic/current_Q2/mean', current_Q2.mean(), self.total_it)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
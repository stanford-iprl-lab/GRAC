import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ES import Searcher
from GRAC_base import GRAC_base
from torch.distributions import Normal

import matplotlib.pyplot as plt
import datetime
import os
# Implementation of Self-Guided and Self-Regularized Actor-Critic Algorithm.

epsilon = 1e-6

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3_sigma = nn.Linear(256, action_dim)
		self.l3_mean = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		self.action_dim = action_dim
		self.state_dim = state_dim

	def forward(self, state, *args):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.max_action * torch.tanh(self.l3_mean(a))
		sigma = (F.softplus(self.l3_sigma(a)) + 0.001).clamp(0.001,2.0*self.max_action)
		normal = Normal(mean, sigma)
		action = normal.rsample().clamp(-self.max_action, self.max_action)
		return action

	def forward_all(self, state, *args):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.max_action * torch.tanh(self.l3_mean(a))
		sigma = (F.softplus(self.l3_sigma(a)) + 0.001).clamp(0.001,2.0*self.max_action)
		normal = Normal(mean, sigma)
		action = normal.rsample()
		log_prob = normal.log_prob(action).sum(1,keepdim=True)
		action = action.clamp(-self.max_action, self.max_action)
		return action, log_prob, mean, sigma

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):

		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def Q2(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q2


class GRAC(GRAC_base):
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
		alpha_start=0.7,
        alpha_end=0.85,
		no_critic_cem=False,
		device=torch.device('cuda'),
	):
		GRAC_base.__init__(self, state_dim, action_dim, max_action, batch_size, discount, tau, policy_noise, noise_clip, policy_freq, device)

		ACTOR_LR  = {
            'Ant-v2': 1e-4,
            'Humanoid-v2': 1e-4,
            'HalfCheetah-v2': 1e-3,
            'Hopper-v2': 2e-4,
            'Swimmer-v2': 2e-4,
            'Walker2d-v2': 2e-4,
		}
		self.actor_lr =  ACTOR_LR[env]

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		
		CRITIC_LR  = {
            'Ant-v2': 3e-4,
            'Humanoid-v2': 3e-4,
            'HalfCheetah-v2': 1e-3,
            'Hopper-v2': 3e-4,
            'Swimmer-v2': 3e-4,
            'Walker2d-v2': 3e-4,
		}
		self.critic_lr =  CRITIC_LR[env]

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

		cem_sigma = 1e-2 * self.max_action * self.max_action
		cem_clip = 0.5 * self.max_action
		self.searcher = Searcher(action_dim, max_action, device=device, sigma_init=cem_sigma, clip=cem_clip, batch_size=batch_size)
		self.action_dim = float(action_dim)
		self.log_freq = 200

		THIRD_LOSS_BOUND = {
             'Ant-v2': 0.75,
             'Humanoid-v2': 0.8,
             'HalfCheetah-v2': 0.85,
             'Hopper-v2': 0.85,
             'Swimmer-v2': 0.6,
             'Walker2d-v2': 0.85,
		}
		self.third_loss_bound = THIRD_LOSS_BOUND[env]

		THIRD_LOSS_BOUND_END = {
             'Ant-v2': 0.85,
             'Humanoid-v2': 0.9,
             'HalfCheetah-v2': 0.9,
             'Hopper-v2': 0.9,
             'Swimmer-v2': 0.8,
             'Walker2d-v2': 0.9,
		}
		self.third_loss_bound_end = THIRD_LOSS_BOUND_END[env]
	
		MAX_TIMESTEPS = {
                        'Ant-v2': 3e6,
                        'Humanoid-v2': 5e6,
                        'HalfCheetah-v2': 3e6,
                        'Hopper-v2': 1e6,
                        'Swimmer-v2': 1e6,
                        'Walker2d-v2': 1e6,
		}
		self.max_timesteps = MAX_TIMESTEPS[env]

		MAX_ITER_STEPS = {
                        'Ant-v2': 100,
                        'Humanoid-v2': 100,
                        'HalfCheetah-v2': 100,
                        'Hopper-v2': 20,
                        'Swimmer-v2': 20,
                        'Walker2d-v2': 20,
		}

		self.max_iter_steps = MAX_ITER_STEPS[env]

		CEM_LOSS_COEF = {
                        'Ant-v2': 1./float(self.action_dim),
                        'Humanoid-v2': 1./float(self.action_dim),
                        'HalfCheetah-v2': 1./float(self.action_dim),
                        'Hopper-v2': 1.0/float(self.action_dim),
                        'Swimmer-v2': 1./float(self.action_dim),
                        'Walker2d-v2': 1.0/float(self.action_dim),
		}

		self.cem_loss_coef = CEM_LOSS_COEF[env]

		EXPL_COEF = {
                        'Ant-v2': 0.01,
                        'Humanoid-v2': 0.01,
                        'HalfCheetah-v2': 0.02,
                        'Hopper-v2': 0.0,
                        'Swimmer-v2': 0.01,
                        'Walker2d-v2': 0.01,
		}
		self.expl_coef = CEM_LOSS_COEF[env]


	def select_action(self, state, writer=None, test=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		if test is False:
			with torch.no_grad():
				action, _ , mean, sigma = self.actor.forward_all(state)
				ceof = 0.95 - min(0.9, float(self.total_it) * 10.0/float(self.max_timesteps))
				writer.add_scalar('train/ceof_select_action',ceof, self.total_it)
				if np.random.uniform(0,1) < ceof:
					better_action = self.searcher.search(state, mean, self.critic.Q2, batch_size=1, cov=sigma**2, sampled_action=action, n_iter=1)
					Q1, Q2 = self.critic(state, action)
					Q = torch.min(Q1, Q2)
					better_Q1, better_Q2 = self.critic(state, better_action)
					better_Q = torch.min(better_Q1, better_Q2)
					action_index = (Q > better_Q).squeeze()
					better_action[action_index] = action[action_index]
				else:
					better_action = action
			return better_action.cpu().data.numpy().flatten()
		else:
			_, _, action, _ = self.actor.forward_all(state)
			return action.cpu().data.numpy().flatten()

	def lr_scheduler(self, optimizer,lr):
		return super().lr_scheduler(optimizer, lr)


	def update_critic(self, critic_loss):
		super().update_critic(critic_loss)

	def train(self, replay_buffer, batch_size=100, writer=None, reward_range=20.0):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():

			# Select action according to policy and add clipped noise
			next_action, _, next_mean, next_sigma = self.actor.forward_all(next_state)
			better_next_action = self.searcher.search(next_state, next_mean, self.critic.Q2, cov=next_sigma**2,sampled_action=next_action,n_iter=1)

			target_Q1, target_Q2 = self.critic(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)

			better_target_Q1, better_target_Q2 = self.critic(next_state, better_next_action)
			better_target_Q = torch.min(better_target_Q1, better_target_Q2)

			action_index = (target_Q > better_target_Q).squeeze()
			if log_it:
				writer.add_scalar('q_diff/BetterQ_smaller_Q',torch.sum(action_index),self.total_it)
				better_Q1_better_Q2_diff = better_target_Q1 - better_target_Q2
				writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_max', better_Q1_better_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_min', better_Q1_better_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_mean', better_Q1_better_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_abs_mean', better_Q1_better_Q2_diff.abs().mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_num', (better_Q1_better_Q2_diff > 0).sum(), self.total_it)

				better_Q1_Q1_diff = better_target_Q1 - target_Q1 
				writer.add_scalar('q_diff_1/better_Q1_Q1_diff_max', better_Q1_Q1_diff.max(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_Q1_diff_min', better_Q1_Q1_diff.min(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_Q1_diff_mean', better_Q1_Q1_diff.mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_Q1_diff_abs_mean', better_Q1_Q1_diff.abs().mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q1_Q1_diff_num', (better_Q1_Q1_diff > 0).sum(), self.total_it)

				better_Q2_Q2_diff = better_target_Q2 - target_Q2 
				writer.add_scalar('q_diff_1/better_Q2_Q2_diff_max', better_Q2_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q2_Q2_diff_min', better_Q2_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q2_Q2_diff_mean', better_Q2_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q2_Q2_diff_abs_mean', better_Q2_Q2_diff.abs().mean(), self.total_it)
				writer.add_scalar('q_diff_1/better_Q2_Q2_diff_num', (better_Q2_Q2_diff > 0).sum(), self.total_it)

			better_next_action[action_index] = next_action[action_index]
			better_target_Q1, better_target_Q2 = self.critic(next_state, better_next_action)

			better_target_Q = torch.max(better_target_Q, target_Q)


			target_Q_final = reward + not_done * self.discount * better_target_Q

			if log_it:
				target_Q1_diff = better_target_Q1 - target_Q1 
				writer.add_scalar('train_critic/target_Q1_diff_max', target_Q1_diff.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q1_diff_mean', target_Q1_diff.mean(), self.total_it)
				writer.add_scalar('train_critic/target_Q1_diff_min', target_Q1_diff.min(), self.total_it)
	
				target_Q2_diff = better_target_Q2 - target_Q2
				writer.add_scalar('train_critic/target_Q2_diff_max', target_Q2_diff.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q2_diff_mean', target_Q2_diff.mean(), self.total_it)
				writer.add_scalar('train_critic/target_Q2_diff_min', target_Q2_diff.min(), self.total_it)
	
				before_target_Q1_Q2_diff = target_Q1 - target_Q2
				writer.add_scalar('q_diff/before_target_Q1_Q2_diff_max', before_target_Q1_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/before_target_Q1_Q2_diff_min', before_target_Q1_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/before_target_Q1_Q2_diff_mean', before_target_Q1_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/before_target_Q1_Q2_diff_abs_mean', before_target_Q1_Q2_diff.abs().mean(), self.total_it)
			
			target_Q1 = better_target_Q1
			target_Q2 = better_target_Q2
			#target_Q1, target_Q2 = self.critic(next_state, better_next_action)
			next_action = better_next_action

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
		self.update_critic(critic_loss)
		critic_loss2 = 0.0

		current_Q1_, current_Q2_ = self.critic(state, action)
		target_Q1_, target_Q2_ = self.critic(next_state, next_action)
		loss3_max_init = torch.pow(current_Q1_ - target_Q_final, 2) + torch.pow(current_Q2_- target_Q_final, 2) + torch.pow(target_Q1_ - target_Q1, 2) + torch.pow(target_Q2_ - target_Q2, 2)
		critic_loss3 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final) + F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)
		self.update_critic(critic_loss3)
		prev_prev_critic_loss3 = critic_loss3.clone() * 1.0 
		init_critic_loss3 = critic_loss.clone()
		ratio = 0.0
		max_step = 0
		writer.add_scalar('train_critic/third_violation_max_loss3_init',torch.max(loss3_max_init), self.total_it)

		idi = 0
		cond1 = 0
		cond2 = 0
		while True:
			idi = idi + 1
			current_Q1_, current_Q2_ = self.critic(state, action)
			target_Q1_, target_Q2_ = self.critic(next_state, next_action)
			loss3_max = torch.pow(current_Q1_ - target_Q_final, 2) + torch.pow(current_Q2_- target_Q_final,2) + torch.pow(target_Q1_- target_Q1, 2) + torch.pow(target_Q2_- target_Q2, 2)
			critic_loss3 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final) + F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)
			self.update_critic(critic_loss3)
			if self.total_it < self.max_timesteps:
				bound = self.third_loss_bound + float(self.total_it) / float(self.max_timesteps) * (self.third_loss_bound_end - self.third_loss_bound)
			else:
				bound = self.third_loss_bound_end
			if critic_loss3 < init_critic_loss3 * bound:# and torch.max(loss3_max) < torch.max(loss3_max_init) * bound:
				cond1 = 1
				break   
			if idi > self.max_iter_steps:
				cond2 = 1
				break
		writer.add_scalar('train_critic/third_loss_num', idi, self.total_it)
		if log_it:
			writer.add_scalar('train_critic/third_loss_cond1', cond1, self.total_it)
			writer.add_scalar('train_critic/third_loss_cond2', cond2, self.total_it)

		if log_it:
			writer.add_scalar("losses/repeat_l1",critic_loss, self.total_it)
			writer.add_scalar("losses/repeat_l3",critic_loss3, self.total_it)
		if log_it:
			after_target_Q1, after_target_Q2 = self.critic(next_state, next_action)
			after_current_Q1, after_current_Q2 = self.critic(state, action)

		#critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
		weights_actor_lr = (abs(current_Q1_ - target_Q_final) + abs(current_Q2_- target_Q_final)).mean().detach()

		if self.total_it % 1 == 0:
			lr_tmp = self.actor_lr / (float(weights_actor_lr)+1.0)
			self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, lr_tmp)

			# Compute actor loss
			actor_action, log_prob, action_mean, action_sigma = self.actor.forward_all(state)
			q_actor_action = self.critic.Q1(state, actor_action)
			m = Normal(action_mean, action_sigma)

			better_action = self.searcher.search(state, action_mean, self.critic.Q1, batch_size=batch_size, cov=(action_sigma)**2, sampled_action=actor_action)
			q_better_action = self.critic.Q1(state, better_action)
			log_prob_better_action = m.log_prob(better_action).sum(1,keepdim=True)

			adv = (q_better_action - q_actor_action).detach()
			adv = torch.max(torch.zeros_like(adv),adv)
			cem_loss = log_prob_better_action * torch.min(reward_range * torch.ones_like(adv),adv)
			if log_it:
				writer.add_scalar('train_actor/cem_loss_ceof', self.cem_loss_coef, self.total_it)
			expl_ceof = self.expl_coef * (1 - float(self.total_it)/self.max_timesteps)
			actor_loss = -(cem_loss * self.cem_loss_coef + q_actor_action + expl_ceof * torch.log(action_sigma).sum(1,keepdim=True)).mean()

			# Optimize the actor 
			Q_before_update = self.critic.Q1(state, actor_action)

			if log_it:
				l1_weight_before = self.actor.l1.weight.clone()
				l3_mean_weight_before = self.actor.l3_mean.weight.clone()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			if log_it:
				l1_grad = self.actor.l1.weight.grad
			self.actor_optimizer.step()

			self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, self.actor_lr)
			
			if log_it:
				Q_after_update = self.critic.Q1(state, self.actor(state))
				prop_improve = 1.0 * torch.sum(Q_after_update >= Q_before_update) / batch_size 
				val_improve = torch.mean(Q_after_update - Q_before_update) 
				val_improve_min = torch.min(Q_after_update - Q_before_update)
				val_improve_max = torch.max(Q_after_update - Q_before_update)

				writer.add_scalar('train_actor/l1_grad_norm', l1_grad.norm(), self.total_it)
				writer.add_scalar('train_actor/l1_grad_norm_weighted', (l1_grad * lr_tmp).norm(), self.total_it)
				with torch.no_grad():
					l1_weight_diff = self.actor.l1.weight - l1_weight_before
					l3_mean_weight_diff = self.actor.l3_mean.weight - l3_mean_weight_before
				writer.add_scalar('train_actor/l1_weight_diff_abs_mean', l1_weight_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_actor/l1_weight_diff_abs_max', l1_weight_diff.abs().max(), self.total_it)
				writer.add_scalar('train_actor/l3_mean_weight_diff_abs_mean', l3_mean_weight_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_actor/l3_mean_weight_diff_abs_max', l3_mean_weight_diff.abs().max(), self.total_it)
	
				writer.add_scalar('train_actor/backtrack_lr', lr_tmp, self.total_it)
				writer.add_scalar('train_actor/actor_loss', actor_loss, self.total_it)
				writer.add_scalar('train_actor/Q_prop_improve', prop_improve, self.total_it)
				writer.add_scalar('train_actor/Q_val_improve', val_improve, self.total_it)
				writer.add_scalar('train_actor/Q_val_improve_min', val_improve_min, self.total_it)
				writer.add_scalar('train_actor/Q_val_improve_max', val_improve_max, self.total_it)
				writer.add_scalar('train_actor/Q_val_improve_std', (Q_after_update - Q_before_update).std(), self.total_it)
				writer.add_scalar('train_actor/log_prob', log_prob.mean(), self.total_it)
				writer.add_scalar('train_actor/action_mean_abs', torch.abs(action_mean).mean(), self.total_it)
				writer.add_scalar('train_actor/action_sigma_abs', torch.abs(action_sigma).mean(), self.total_it)
	
				# cem logging
				writer.add_scalar('cem/log_prob_better_action_mean', log_prob_better_action.mean(), self.total_it)
				writer.add_scalar('cem/log_prob_better_action_max', log_prob_better_action.max(), self.total_it)
				writer.add_scalar('cem/log_prob_better_action_min', log_prob_better_action.min(), self.total_it)
				writer.add_scalar('cem/log_prob_better_action_std', log_prob_better_action.std(), self.total_it)
				writer.add_scalar('cem/q_better_action_mean', q_better_action.mean(), self.total_it)
				writer.add_scalar('cem/q_better_action_max', q_better_action.max(), self.total_it)
				writer.add_scalar('cem/q_better_action_min', q_better_action.min(), self.total_it)
				writer.add_scalar('cem/q_better_action_std', q_better_action.std(), self.total_it)
				writer.add_scalar('cem/q_diff_mean', (q_better_action - q_actor_action).mean(), self.total_it)
				writer.add_scalar('cem/q_diff_max', (q_better_action - q_actor_action).max(), self.total_it)
				writer.add_scalar('cem/q_diff_min', (q_better_action - q_actor_action).min(), self.total_it)
				writer.add_scalar('cem/q_diff_std', (q_better_action - q_actor_action).std(), self.total_it)
				writer.add_scalar('cem/better_action_diff', (better_action - actor_action).abs().mean(), self.total_it)
				writer.add_scalar('cem/cem_loss_mean', (cem_loss).mean(), self.total_it)
				writer.add_scalar('cem/cem_loss_max', (cem_loss).max(), self.total_it)
				writer.add_scalar('cem/cem_loss_min', (cem_loss).min(), self.total_it)
				writer.add_scalar('cem/cem_loss_std', (cem_loss).std(), self.total_it)
				writer.add_scalar('cem/v_mean', q_actor_action.mean(), self.total_it)
				writer.add_scalar('cem/v_max', q_actor_action.max(), self.total_it)
				writer.add_scalar('cem/v_min', q_actor_action.min(), self.total_it)
				writer.add_scalar('cem/v_std', q_actor_action.std(), self.total_it)

		if log_it:
			with torch.no_grad():
				writer.add_scalar('train_critic/critic_loss', critic_loss, self.total_it)
				writer.add_scalar('losses/critic_loss2', critic_loss2, self.total_it)
				writer.add_scalar('losses/critic_loss3', critic_loss3, self.total_it)
	
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
				writer.add_scalar('q_diff/target_Q1_Q2_diff_num', (target_Q1_Q2_diff > 0).sum(), self.total_it)
	
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
	
				done = 1 - not_done
				writer.add_scalar('losses/done_max', done.max(), self.total_it)
				writer.add_scalar('losses/done_min', done.min(), self.total_it)
				writer.add_scalar('losses/done_mean', done.mean(), self.total_it)
				
				
				#target_Q1
				writer.add_scalar('train_critic/target_Q1/mean', torch.mean(target_Q1), self.total_it)
				writer.add_scalar('train_critic/target_Q1/max', target_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/min', target_Q1.min(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/std', torch.std(target_Q1), self.total_it)
				writer.add_scalar('train_critic/target_Q1_after/mean', torch.mean(after_target_Q1), self.total_it)

				target_Q1_update_diff = after_target_Q1 - target_Q1
				writer.add_scalar('train_critic_update_diff/target_Q1_update_diff_mean', target_Q1_update_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q1_update_diff_max', target_Q1_update_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q1_update_diff_min', target_Q1_update_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q1_update_diff_abs_mean', target_Q1_update_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q1_update_diff_std', target_Q1_update_diff.std(), self.total_it)
	
				#target_Q2
				writer.add_scalar('train_critic/target_Q2/mean', torch.mean(target_Q2), self.total_it)
				target_Q2_update_diff = after_target_Q2 - target_Q2
				writer.add_scalar('train_critic_update_diff/target_Q2_update_diff_mean', target_Q2_update_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q2_update_diff_max', target_Q2_update_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q2_update_diff_min', target_Q2_update_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q2_update_diff_abs_mean', target_Q2_update_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/target_Q2_update_diff_std', target_Q2_update_diff.std(), self.total_it)
	
				#current_Q1
				writer.add_scalar('train_critic/current_Q1/mean', current_Q1.mean(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/std', torch.std(current_Q1), self.total_it)
				writer.add_scalar('train_critic/current_Q1_after/mean', torch.mean(after_current_Q1), self.total_it)
				writer.add_scalar('train_critic/current_Q1/max', current_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/min', current_Q1.min(), self.total_it)
	
				current_Q1_update_diff = after_current_Q1 - current_Q1
				writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_mean', current_Q1_update_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_max', current_Q1_update_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_min', current_Q1_update_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_abs_mean', current_Q1_update_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_std', current_Q1_update_diff.std(), self.total_it)
	
				# current_Q2
				writer.add_scalar('train_critic/current_Q2/mean', current_Q2.mean(), self.total_it)
				current_Q2_update_diff = after_current_Q2 - current_Q2
				writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_mean', current_Q2_update_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_max', current_Q2_update_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_min', current_Q2_update_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_abs_mean', current_Q2_update_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_std', current_Q2_update_diff.std(), self.total_it)
	
				current_Q1_goal_diff = target_Q_final - after_current_Q1
				writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_mean', current_Q1_goal_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_max', current_Q1_goal_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_min', current_Q1_goal_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_abs_mean', current_Q1_goal_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_std', current_Q1_goal_diff.std(), self.total_it)
	
				current_Q2_goal_diff = target_Q_final - after_current_Q2
				writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_mean', current_Q2_goal_diff.mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_max', current_Q2_goal_diff.max(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_min', current_Q2_goal_diff.min(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_abs_mean', current_Q2_goal_diff.abs().mean(), self.total_it)
				writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_std', current_Q2_goal_diff.std(), self.total_it)

	def save(self, filename):
		super().save(filename)

	def load(self, filename):
		super().load(filename)

	def make_Q_contour(self, state, save_folder, base_action):
		super().make_Q_contour(state, save_folder, base_action)

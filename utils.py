import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
class ReplayBufferTorch(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, state_dim), device=device)
		self.action = torch.zeros((max_size, action_dim), device=device)
		self.next_state = torch.zeros((max_size, state_dim), device=device)
		self.reward = torch.zeros((max_size, 1), device=device)
		self.not_done = torch.zeros((max_size, 1), device=device)

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = torch.tensor(state, device=self.device)
		self.action[self.ptr] = torch.tensor(action, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)
	# def save(self, log_dir):
	# 	import pickle
	# 	with open('{}/buffer_{}.pkl'.format(log_dir, self.ptr), 'wb+') as f:
	# 		pickle.dump({
	# 			'state': self.state,
	# 			'action': self.action,
	# 			'next_state': self.next_state,
	# 			'reward': self.reward,
	# 			'not_done': self.not_done
	# 		}, f)

class WriterLoggerWrapper(object):
	def __init__(self, log_dir, comment, max_timesteps):
		self.tf_writer = SummaryWriter(log_dir=log_dir, comment=comment)
		
		logger_result_path = '{}/{}'.format(log_dir, 'log_txt')
		if not os.path.exists(logger_result_path):
			os.makedirs(logger_result_path)
		print(logger_result_path)
		self.logger = Logger(logger_result_path, max_timesteps)

	def add_scalar(self, scalar_name, scalar_val, it):
		self.tf_writer.add_scalar(scalar_name, scalar_val, it)
		self.logger.add_scalar(scalar_name, scalar_val, it)

class Logger(object):
	def __init__(self, log_dir, max_timesteps):
		self.log_dir = log_dir
		self.max_timesteps = max_timesteps
		self.all_data = {}

	def add_scalar(self, scalar_name, scalar_val, it):
		if not (scalar_name in self.all_data.keys()):
			# add new entry
			self.all_data[scalar_name] = np.zeros([int(self.max_timesteps + 1)])
		self.all_data[scalar_name][int(it)] = scalar_val
	
	def save_to_txt(self, log_dir=None):
		if log_dir is None:
			log_dir = self.log_dir
		
		for tag in self.all_data.keys():
			np.savetxt('{}/{}data.txt'.format(log_dir, tag.replace('/', '_')), self.all_data[tag], delimiter='\n', fmt='%.5f')

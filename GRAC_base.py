import torch

class GRAC_base(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		batch_size,
		discount,
		tau,
		policy_noise,
		noise_clip,
		policy_freq,
		device,
	):
		self.action_dim = action_dim
		self.state_dim = state_dim

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		print('device', device, policy_freq)
		self.device = device

	def log_all(self, writer, label, v, total_it):
		with torch.no_grad():
			writer.add_scalar('{}_mean'.format(label), v.mean(), total_it)
			writer.add_scalar('{}_max'.format(label), v.max(), total_it)
			writer.add_scalar('{}_min'.format(label), v.min(), total_it)
			writer.add_scalar('{}_abs_mean'.format(label), v.abs().mean(), total_it)
			writer.add_scalar('{}_std'.format(label), v.std(), total_it)

	def select_action(self, state, writer=None, test=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		return action.cpu().data.numpy().flatten()

	def lr_scheduler(self, optimizer,lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return optimizer

	def update_critic(self, critic_loss):
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


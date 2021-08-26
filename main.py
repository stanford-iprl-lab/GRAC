import numpy as np
import torch
import gym
import argparse
import os

import utils
import datetime

from torch.utils.tensorboard import SummaryWriter
from sliding_window import SlidingMin 
from RollerGrasperV2 import robot_env 

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, args=None):

	if env_name == 'RollerGrasperV2':
		eval_env = robot_env.RobotEnv(args)
	else:
		eval_env = gym.make(env_name)
		eval_env.seed(seed + 100)

	avg_reward = 0.
	avg_succ = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), test=True)
			state, reward, done, succ = eval_env.step(action)
			avg_reward += reward

			if type(succ) == float:
				avg_succ += succ
		
	avg_reward /= eval_episodes
	avg_succ /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over {} episodes: {:.3f} {:.3f}".format(eval_episodes, avg_reward, avg_succ))
	print("---------------------------------------")
	return avg_reward, avg_succ


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="GRAC")                  # Policy name (GRAC)
	parser.add_argument("--env", default="Ant-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=3e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument('--n_repeat', default=10, type=int)
	parser.add_argument('--alpha_start', default=0.7,type=float)
	parser.add_argument('--alpha_end', default=0.85,type=float)
	parser.add_argument('--use_expl_noise', action="store_true")
	parser.add_argument('--no_adaptive_lr', action="store_true")
	parser.add_argument('--adaptive_lr_weight', default=1, type=float)
	parser.add_argument('--model_noise', default=0, type=float)


	parser.add_argument('--linear_reward', action="store_true")
	parser.add_argument('--expert_demo', action="store_true")


	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--comment", default="")
	parser.add_argument("--exp_name", default="exp_ant")
	parser.add_argument("--which_cuda", default=0, type=int)

	args = parser.parse_args()

	device = torch.device('cuda:{}'.format(args.which_cuda))

	file_name = "{}_{}_{}".format(args.policy, args.env, args.seed)
	file_name += "_{}".format(args.comment) if args.comment != "" else ""
	folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name
	result_folder = 'runs/{}'.format(folder_name) 
	if args.exp_name == "":
		result_folder = '{}/{}'.format(args.exp_name, folder_name)
	if args.debug: 
		result_folder = 'debug/{}'.format(folder_name)

	if not os.path.exists('{}/models/'.format(result_folder)):
		os.makedirs('{}/models/'.format(result_folder))
	print("---------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")



	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	if args.env == 'RollerGrasperV2':
		env = robot_env.RobotEnv(args)
		state_dim = env.state_dim
		action_dim = env.action_dim
		max_action = env.max_action
		# env.reset(target_axis=np.array([0, 0, 1]))
		# assert False

		args.save_model = True
		
	else:
		env = gym.make(args.env)
		env.seed(args.seed)
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

	if args.save_model is False:
		args.save_model = True
	kwargs = {
		"env": args.env,
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": torch.from_numpy(max_action).type(torch.FloatTensor).to(device) if args.env == 'RollerGrasperV2' else max_action,
		"batch_size": args.batch_size,
		"discount": args.discount,
		"tau": args.tau,
		"n_repeat": args.n_repeat,
		"alpha_start": args.alpha_start,
		"alpha_end":args.alpha_end,
		"device": device,
	}
	if args.policy == 'GRAC_clipq':
		kwargs["no_adaptive_lr"] = args.no_adaptive_lr
		kwargs["adaptive_lr_weight"] = args.adaptive_lr_weight
	if args.policy == 'GRAC_noise' or 'noise' in args.policy:
		kwargs['model_noise'] = args.model_noise
		
	# Initialize policy
	if "GRAC" in args.policy or 'TD3' in args.policy:
		# Target policy smoothing is scaled wrt the action scale
		GRAC = __import__(args.policy)
		policy = GRAC.GRAC(**kwargs)


	if args.load_model != "":
		policy_file = 'model' if args.load_model == "default" else args.load_model
		policy.load("./{}/{}".format(result_folder, policy_file))

	replay_buffer = utils.ReplayBufferTorch(state_dim, action_dim, device=device)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, args=args)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	reward_max = 1.0
	reward_min = np.inf
	episode_step_max = 1
	episode_step_min = 1000

	reward_min_buffer = SlidingMin(int(1e5))
	episode_step_min_buffer = SlidingMin(int(1e3))
	episode_step_min_buffer.insert(1000)

	# writer = utils.WriterLoggerWrapper(result_folder, comment=file_name, max_timesteps=args.max_timesteps)
	writer = SummaryWriter(log_dir=result_folder, comment=file_name)

	#record all parameters value
	with open("{}/parameters.txt".format(result_folder), 'w') as file:
		for key, value in vars(args).items():
			file.write("{} = {}\n".format(key, value))

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			if args.expert_demo:
				action = env.get_expert_action()
			else:
				action = np.random.uniform(-max_action, max_action, action_dim)
		else:
			if args.use_expl_noise:
				action = (
					policy.select_action(np.array(state),writer=writer) 
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)
			else:
				action = (
					policy.select_action(np.array(state),writer=writer)
				).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		if reward > reward_max:
			reward_max = reward
		reward_min_buffer.insert(reward)
		 
		writer.add_scalar('test/reward', reward, t+1)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		reward_kwargs = {
			'reward_max': reward_max, 
			'episode_step_max': episode_step_max, 
			'reward_min': reward_min_buffer.get_min(), 
			'episode_step_min': episode_step_min_buffer.get_min()
		}
		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if args.policy == 'GRAC_clipq' or 'GRAC_clipq' in args.policy:
				policy.train(replay_buffer, args.batch_size, writer, 20.0, **reward_kwargs)
			else:
				policy.train(replay_buffer, args.batch_size, writer, 20.0)#reward_max - reward_min)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
			if episode_timesteps > episode_step_max:
				episode_step_max = episode_timesteps
			episode_step_min_buffer.insert(episode_timesteps)

			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluation, evaluation_succ = eval_policy(policy, args.env, args.seed, args=args)
			evaluations.append(evaluation)
			print('hello')
			writer.add_scalar('test/avg_return', evaluation, t+1)
			writer.add_scalar('test/avg_succ', evaluation_succ, t+1)
			np.save("{}/evaluations".format(result_folder), evaluations)

		if (t + 1) % 50000 == 0: 
			if args.save_model:
				policy.save("./{}/models/iter_{}_model".format(result_folder, t + 1))
			# replay_buffer.save(result_folder)
		
		# save to txt
		# if (t + 1) % 50000 == 0:
			# writer.logger.save_to_txt()

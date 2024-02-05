#!/usr/bin/env python3
from stable_baselines3 import PPO, DDPG
import os
from drone_env import DroneEnv
import time



models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = DroneEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model = DDPG(
#             "MlpPolicy",  
#             env,
#             buffer_size=10000,  # Replay buffer size
#             learning_rate=1e-3,  # Learning rate for the actor and critic networks
#             batch_size=64,  # Batch size for training
#             gamma=0.99,  # Discount factor for future rewards
#             tau=0.005,  # Soft update coefficient for target networks
#             policy_kwargs=dict(net_arch=[400, 300]),  # Architecture of the policy network
#             action_noise=env.action_noise,
#             verbose=1, tensorboard_log=logdir
# )

TIMESTEPS = 10000
iters = 0
for _ in range(10000):
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
#!/usr/bin/env python3
from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
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
# env.reset()

model = PPO("CnnPolicy", env,verbose=1, n_steps = 10000, learning_rate = 0.0001, tensorboard_log=logdir)

TIMESTEPS = 1e7
iters = 0

while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	

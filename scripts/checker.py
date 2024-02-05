#!/usr/bin/env python3
from drone_env import DroneEnv

env = DroneEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		# print("action",random_action)
		obs, reward, done, info = env.step(random_action)
		# print('reward',reward)
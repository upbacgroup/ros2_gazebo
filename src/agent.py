#!/usr/bin/env python3

import rospy
import gym
from custom_env import DroneNavigationEnv
from train import TrainAgent
import numpy as np
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import random
from stable_baselines3 import PPO

rospy.init_node('drone_agent')


env = DroneNavigationEnv()
env.observation = env.reset()
agent = TrainAgent()
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./board/")

done = False

while not done:
    # if env.current_depth_map is not None:
        # env.observation = env.current_depth_map[:, :, np.newaxis]  # Update the observation first
        env.action, _ = model.predict(env.get_observations())  # Predict action based on the updated observation
        env.observation, env.reward, env.done, _ = env.step(env.action)

    #     env.observation = np.zeros((env.width, env.height, 1), dtype=np.float32)  # Default value    























#env.close()

"""# Import necessary libraries and modules
import numpy as np

# Initialize environment, PPO agent, and other necessary components
env = initialize_environment()  # Initialize your ROS and Gazebo environment
agent = initialize_ppo_agent()  # Initialize your PPO agent with appropriate configurations
state = env.reset()  # Reset the environment and obtain initial state

# Define training parameters
num_episodes = 1000  # Number of episodes for training
max_steps_per_episode = 1000  # Maximum number of steps per episode

# Training loop
for episode in range(num_episodes):
    episode_reward = 0  # Initialize episode reward
    state = env.reset()  # Reset the environment and obtain initial state
    
    for step in range(max_steps_per_episode):
        # Select action based on the current state using the PPO agent
        action = agent.select_action(state)
        
        # Execute selected action in the environment and obtain next state, reward, done flag, and additional info
        next_state, reward, done, _ = env.step(action)
        
        # Store experience (state, action, reward, next_state, done) in agent's memory
        agent.store_experience(state, action, reward, next_state, done)
        
        # Update current state
        state = next_state
        
        # Accumulate episode reward
        episode_reward += reward
        
        # Perform PPO agent's update step if memory size reaches a certain threshold or episode terminates
        if len(agent.memory) >= agent.batch_size or done:
            agent.update()  # Perform PPO agent's update step
        
        # Break if episode is done
        if done:
            break
    
    # Print episode information
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward}")
    
    # Save model checkpoints, logs, or other necessary information based on your requirements

# Save final trained PPO agent model
agent.save_model("ppo_agent_model.pth")
"""

    
#!/usr/bin/env python3
from os import times
from tabnanny import verbose
from turtle import distance
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist

import rospy
import numpy as np
from tf.transformations import *
from air_drone.msg import Pose, MotorSpeed
import gym
from gym import spaces
import rospy
from sensor_msgs.msg import Image
import cv2
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise

from std_srvs.srv import Empty
from rospy.exceptions import ROSTimeMovedBackwardsException
import random
import math
import time
import tensorflow as tf
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from ddpg import DroneNavigationEnv

if __name__ == '__main__':
    rospy.init_node('ddpg')
    env = DroneNavigationEnv()
    rate = rospy.Rate(1)
    num_episodes = 1
    total_timesteps_ = 1000000
    total_timesteps = 0
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/")
    print("------------Starting Learning------------")
    model.learn(total_timesteps=total_timesteps_)
    print("------------Done Learning------------")
    env.generate_goal()
    env.observation = env.reset()
    done = False
        
    while not rospy.is_shutdown(): 
        # print(f"Timestep {timestep + 1}/{total_timesteps_}")
        if env.current_depth_map is not None:
            env.observation = env.current_depth_map[:, :, np.newaxis]  # Update the observation first
            env.action, _ = model.predict(env.observation)  # Predict action based on the updated observation
            env.observation, env.reward, env.done, env.info = env.step(env.action)
           

        else:
            env.observation = np.zeros((env.width, env.height, 1), dtype=np.float32)  # Default value

        if done:
            print("Done, collected", env.collected_target_points, "target points")
            env.reset()
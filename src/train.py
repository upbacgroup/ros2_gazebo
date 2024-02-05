#!/usr/bin/env python3

from genericpath import exists
import rospy
import numpy as np
from custom_env import DroneNavigationEnv  # Importing the environment from custom_env.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

import os

class TrainAgent():
    def __init__(self):
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

    def main(self):
        env = DroneNavigationEnv()
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./board/")
        #model = PPO.load("", env_env) 

        print("------------Starting Learning------------")

        model.learn(total_timesteps=1000000)
        model.save("drone_env")  
        print("------------Done Learning------------")

if __name__ == '__main__':
    rospy.init_node('train_agent')
    classname = TrainAgent()
    classname.main()


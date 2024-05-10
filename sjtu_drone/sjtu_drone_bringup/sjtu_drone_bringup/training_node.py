# !/usr/bin/env python3
# from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
# import os
# from sjtu_drone_bringup.drone_env import DroneEnv
# import time


# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"

# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)

# env = DroneEnv()
# env.reset()

# model = PPO("CnnPolicy", env,verbose=1, n_steps = 10000, learning_rate = 0.0001, tensorboard_log=logdir)

# TIMESTEPS = 1e7
# iters = 0

# while True:
# 	iters += 1
# 	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
# 	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	
import rclpy
from rclpy.node import Node

from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
import os
from sjtu_drone_bringup.drone_env import DroneEnv
import time


class TrainingNode(Node):

    def __init__(self):
        super().__init__('training_node')

        # (Optional) ROS 2 Subscription logic here
        print("********TRAINING NODE HAS STARTED**************")

        # Define your model and training parameters
        self.models_dir = f"models/{int(time.time())}/"
        self.logdir = f"logs/{int(time.time())}/"
        self.env = DroneEnv()
        # self.env.reset()
        self.model = PPO("CnnPolicy", self.env, verbose=1, tensorboard_log=self.logdir)
        self.TIMESTEPS = 1e7

        # Trigger training on node startup (replace with your desired trigger)
        self.start_training()

    def start_training(self):
        iters = 0
        while True:
            iters += 1
            self.model.learn(total_timesteps=self.TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
            # self.model.save(f"{self.models_dir}/{self.TIMESTEPS*iters}")

def main(args=None):
    rclpy.init(args=args)
    node = TrainingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

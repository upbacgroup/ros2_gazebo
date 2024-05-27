from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
import os
from drone_env import DroneEnv
import time
import tensorflow as tf

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = DroneEnv()
# env = make_vec_env(lambda: env, n_envs=1)
# env = DummyVecEnv([lambda: env])
# env.reset()


# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set the visible GPU devices
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU enabled")
    except RuntimeError as e:
        print(e)

# model = A2C("CnnPolicy", env, learning_rate = 0.001, gamma = 0.95, use_rms_prop=False, verbose=1, tensorboard_log=logdir)
# model = A2C("CnnPolicy", env, learning_rate = 0.001, gamma = 0.95, verbose=1, tensorboard_log=logdir)
model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-7,  tensorboard_log=logdir)

TIMESTEPS = 1e6
iters = 0

while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")     
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
      


    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
         
	


from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
import os
from drone_env import DroneEnv
import time
import tensorflow as tf
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np



models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

# saved_model_path = "path/to/your/saved_model.zip"

# def load_model(model_path, env):
#     return PPO.load(model_path, env=env)



if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = DroneEnv()


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




n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))




# model = A2C("CnnPolicy", env, learning_rate = 0.001, gamma = 0.95, use_rms_prop=False, verbose=1, tensorboard_log=logdir)
# model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-6, gamma = 0.95,  tensorboard_log=logdir)
model = DDPG("CnnPolicy", env, verbose=1, action_noise=action_noise, learning_rate=1e-7, gamma = 0.95,  tensorboard_log=logdir)
 

TIMESTEPS = 1e6
MAX_ITERS = 1e7  # Number of iterations
iters = 0

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DDPG")

# while iters < MAX_ITERS:
#     iters += 1
#     print(f"Starting iteration {iters} out of {MAX_ITERS}")
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DDPG")
#     model.save(f"{models_dir}/{TIMESTEPS*iters}")
#     print(f"Completed iteration {iters}")

# print("Training complete")



# model = load_model(saved_model_path, env)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")


# # Example of using the loaded model for prediction
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)      
      



         
	


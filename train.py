import os
import gymnasium as gym
import numpy as np
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime

for i in range(4):
    for reward_type in ['sparse', 'dense']: 
        # Get current time
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')

        #reward_type = "sparse" # choose 'sparse' or 'dense'
        nsteps = 100000
        n_eval_episodes=50

        if reward_type == 'dense': env_id = 'FetchReachDense-v2'
        else: env_id = 'FetchReach-v2'
        env = gym.make(env_id)
        env = Monitor(env)  # Wrap the environment with the Monitor wrapper
        env = DummyVecEnv([lambda: env])

        # Define a callback for evaluation
        log_dir = os.path.join(os.getcwd(), "logs")


        os.makedirs(log_dir, exist_ok=True)
        eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                     log_path=log_dir, eval_freq=500,
                                     n_eval_episodes=n_eval_episodes,
                                     deterministic=True, render=False)

        # Initialize the model
        model = PPO("MultiInputPolicy", env, verbose=1, device='cuda')

        # Train the model
        model.learn(total_timesteps=nsteps, callback=eval_callback)

        # After training is done, rename or move the evaluations.npz
        original_file_path = os.path.join(log_dir, "evaluations.npz")
        new_file_path = os.path.join(log_dir, f"{reward_type}_evaluations_name_{current_time}.npz")

        # Check if the original file exists and then move/rename it
        if os.path.exists(original_file_path):
            shutil.move(original_file_path, new_file_path)

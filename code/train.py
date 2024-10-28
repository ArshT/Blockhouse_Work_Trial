import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

from code.benchmark import Benchmark
from code.env import TradeExecutionEnv

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def train_func(args):
    # Create the environment
    env = TradeExecutionEnv('data/merged_bid_ask_ohlcv_data.csv')

    # Create a directory to save models
    models_dir = "models/DQN"
    os.makedirs(models_dir, exist_ok=True)

    
    # Define the DQN model using an MLP to represent the Q-network
    # model = DQN(
    #     policy='MlpPolicy',
    #     env=env,
    #     learning_rate=1e-4,
    #     buffer_size=50000,
    #     learning_starts=1000,
    #     batch_size=32,
    #     tau=1.0,
    #     gamma=0.99,
    #     train_freq=4,
    #     target_update_interval=1000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     verbose=1,
    #     tensorboard_log="./tensorboard/"
    # )

    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    # Define a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq
                                             , save_path=models_dir, 
                                             name_prefix='dqn_model')

    # Train the agent
    total_timesteps = args.total_timesteps
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save the final model
    model.save(f"{models_dir}/final_model")


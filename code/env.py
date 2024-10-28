import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from code.benchmark import Benchmark

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback



class TradeExecutionEnv(gym.Env):
    def __init__(self, csv_path, total_shares=1000, trading_horizon=390, tick_size=0.1, I=10, T=10):
        super(TradeExecutionEnv, self).__init__()

        # Load the CSV data
        self.df = pd.read_csv(csv_path)
        print("Length of data: ", len(self.df))

        self.bm = Benchmark(self.df)
        
        # Environment parameters
        self.total_shares = total_shares
        self.trading_horizon = trading_horizon
        self.tick_size = tick_size
        self.I = I  # Inventory resolution (10 levels)
        self.T = T  # Time resolution (10 levels)

        # Calculate inventory and time level sizes
        self.inventory_level_size = total_shares // self.I
        self.time_level_size = trading_horizon // self.T

        # Define observation space (10 levels for inventory, 10 levels for time)
        self.observation_space = spaces.MultiDiscrete([self.I, self.T])

        # Define action space:
        self.action_space = spaces.Discrete(50)  # Actions: 50 possible price adjustments

        # Initial state variables
        self.remaining_shares = total_shares
        self.time_step = 0
        self.ask_price_1 = self.df['ask_price_1'][self.time_step]

        self.num_ep_completed = 0

        

    def get_inventory_level(self):
        # Determine the current inventory level based on remaining shares
        return min(self.remaining_shares // self.inventory_level_size, self.I - 1)

    def get_time_level(self):
        # Determine the current time level based on elapsed time
        return min(self.time_step // self.time_level_size, self.T - 1)

    def reset(self, seed=None, options=None):
        # Reset environment state at the start of each episode
        self.remaining_shares = self.total_shares
        self.ask_price_1 = self.df['ask_price_1'][self.time_step]

        self.time_step = (390 * self.num_ep_completed) % 16000
        # print(self.time_step)
        
        # Return initial observation and info (for Gymnasium compatibility)
        return np.array([self.get_inventory_level(), self.get_time_level()]), {}

    def step(self, action):
        # Adjusted ask price based on price adjustment action
        price_adjustment = (action - 25) * self.tick_size 
        adjusted_ask_price = self.ask_price_1 + price_adjustment

        # Set a fixed number of shares to sell (e.g., based on inventory level)
        shares_to_sell = min(self.total_shares, self.remaining_shares)

        # Calculate transaction cost (total cost incurred)
        transaction_cost = 0
        shares_sold = 0

        # Iterate over bid prices and sizes to execute the trade
        for i in range(1, 6):  # Assuming we have 5 bid levels
            bid_price = self.df[f'bid_price_{i}'][self.time_step]
            bid_size = self.df[f'bid_size_{i}'][self.time_step]

            if adjusted_ask_price <= bid_price:
                # Execute trade at this bid level
                trade_size = min(bid_size, shares_to_sell - shares_sold)  # How much we can sell at this level
                # transaction_cost += (trade_size / 1000) * bid_price

                shares_sold += trade_size

            
            # Stop if we've sold all intended shares for this time step
            if shares_sold >= shares_to_sell:
                break

        # Update remaining shares after the trade
        self.remaining_shares -= shares_sold
        

        ### Reward Function-1

        # # Calculate reward: if no shares were sold, apply a penalty
        # if shares_sold == 0:
        #     reward = -5000
        # else:
        #     reward = -transaction_cost  # Negative reward to minimize cost

        
        ### Reward Function-2 (As mentioned in the task description)
        alpha = 0.01  # Example value; adjust based on empirical data
        shares = shares_sold  # Number of shares sold in this step

        Slippage, Market_Impact = self.bm.compute_components(alpha, shares, self.time_step)
        reward = - (Slippage + Market_Impact)


        # Update environment state
        self.time_step += 1

        # Update ask price based on the current time step in the CSV data
        if self.time_step < self.trading_horizon:
            self.ask_price_1 = self.df['ask_price_1'][self.time_step]

        # Check if episode is done
        done = (self.time_step % self.trading_horizon >= (self.trading_horizon - 1) or self.remaining_shares <= 0)
        truncated = False  # For Gymnasium, you typically set truncated if you have additional stopping criteria

        if done:
            self.num_ep_completed += 1

        # Next observation
        next_state = np.array([self.get_inventory_level(), self.get_time_level()])

        return next_state, reward, done, truncated, {}

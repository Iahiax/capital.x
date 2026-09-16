# rl_env.py

import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.idx = 0

        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self):
        self.idx = 0
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False

        if action == 0:  # Buy
            reward = self.df['Close'].iloc[self.idx + 1] - self.df['Close'].iloc[self.idx]
        elif action == 1:  # Sell
            reward = self.df['Close'].iloc[self.idx] - self.df['Close'].iloc[self.idx + 1]

        self.idx += 1
        if self.idx >= len(self.df) - 2:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        return np.array([
            row['Close'],
            row['EMA_20'],
            row['EMA_100'],
            row['ATR'],
            row['Regime'],
            row['MarketQuality'],
            row['Score'],
            row['RVOL'],
            row['ShockIndex'],
            row['NoiseIndex']
        ], dtype=np.float32)

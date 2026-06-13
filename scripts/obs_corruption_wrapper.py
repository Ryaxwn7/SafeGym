import gymnasium as gym
import numpy as np


class LidarBlindnessWrapper(gym.Wrapper):
    """Mask lidar-like observation tail dimensions to simulate missed danger perception."""

    def __init__(self, env, blind_dims=8, fill_value=1.0):
        super().__init__(env)
        self.blind_dims = int(blind_dims)
        self.fill_value = float(fill_value)

    def _corrupt(self, observation):
        obs = np.array(observation, copy=True)
        if obs.ndim == 1 and obs.shape[0] >= self.blind_dims:
            obs[-self.blind_dims :] = self.fill_value
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._corrupt(obs), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return self._corrupt(obs), reward, cost, terminated, truncated, info

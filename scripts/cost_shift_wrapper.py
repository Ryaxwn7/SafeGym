import gymnasium as gym


class CostUnderestimateWrapper(gym.Wrapper):
    """Expose underestimated cost while keeping true_cost in info."""

    def __init__(self, env, scale=0.25):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        true_cost = float(cost)
        exposed_cost = true_cost * self.scale
        info = dict(info)
        info['true_cost'] = true_cost
        info['exposed_cost'] = exposed_cost
        return obs, reward, exposed_cost, terminated, truncated, info

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


class SafetyGymnasiumPPOEnv(gym.Env):
    """Adapt Safety-Gymnasium's 6-value step API to Gymnasium's 5-value API."""

    metadata = {'render_modes': ['rgb_array', 'human']}

    def __init__(self, env_id, cost_penalty=1.0):
        super().__init__()
        self.env = safety_gymnasium.make(env_id)
        self.cost_penalty = float(cost_penalty)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low.astype(np.float32),
            high=self.env.observation_space.high.astype(np.float32),
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        true_reward = float(reward)
        true_cost = float(cost)
        training_reward = true_reward - self.cost_penalty * true_cost
        info = dict(info)
        info['true_reward'] = true_reward
        info['true_cost'] = true_cost
        info['training_reward'] = training_reward
        return np.asarray(obs, dtype=np.float32), training_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(env_id, cost_penalty, seed):
    env = SafetyGymnasiumPPOEnv(env_id, cost_penalty=cost_penalty)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_model(model, env_id, episodes, seed_start, out_path):
    env = SafetyGymnasiumPPOEnv(env_id, cost_penalty=0.0)
    rows = []

    for episode in range(episodes):
        seed = seed_start + episode
        obs, info = env.reset(seed=seed)
        total_return = 0.0
        total_cost = 0.0
        steps = 0

        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(info.get('true_reward', reward))
            total_cost += float(info.get('true_cost', 0.0))
            steps += 1

            if terminated or truncated:
                break

        rows.append(
            {
                'seed': seed,
                'return': total_return,
                'cost': total_cost,
                'true_cost': total_cost,
                'exposed_cost': total_cost,
                'steps': steps,
                'violated': int(total_cost > 0.0),
            }
        )

    env.close()
    df = pd.DataFrame(
        rows,
        columns=['seed', 'return', 'cost', 'true_cost', 'exposed_cost', 'steps', 'violated'],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointGoal1-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--cost-penalty', type=float, default=1.0)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--model-out', default='models/ppo_safety_point_goal.zip')
    parser.add_argument('--eval-out', default='results/2026-06-13-ppo-baseline/ppo_eval.csv')
    parser.add_argument('--tensorboard-log', default='')
    args = parser.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    set_random_seed(args.seed)

    model_path = Path(args.model_out)
    eval_path = Path(args.eval_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env, args.cost_penalty, args.seed)
    model = PPO(
        'MlpPolicy',
        env,
        seed=args.seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=args.tensorboard_log or None,
        device='cpu',
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(model_path)
    env.close()

    eval_df = evaluate_model(model, args.env, args.eval_episodes, args.seed, eval_path)
    print('model:', model_path)
    print('eval_csv:', eval_path)
    print('avg_return:', float(eval_df['return'].mean()))
    print('avg_true_cost:', float(eval_df['true_cost'].mean()))
    print('violation_rate:', float(eval_df['violated'].mean()))


if __name__ == '__main__':
    main()

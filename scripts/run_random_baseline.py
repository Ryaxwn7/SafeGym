import argparse
from pathlib import Path

import pandas as pd
import safety_gymnasium
from tqdm import trange

from cost_shift_wrapper import CostUnderestimateWrapper
from obs_corruption_wrapper import LidarBlindnessWrapper


def run_episode(env, seed):
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_return = 0.0
    total_cost = 0.0
    total_true_cost = 0.0
    total_exposed_cost = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        true_cost = float(info.get('true_cost', cost))
        exposed_cost = float(info.get('exposed_cost', cost))
        total_return += float(reward)
        total_cost += float(cost)
        total_true_cost += true_cost
        total_exposed_cost += exposed_cost
        steps += 1

        if terminated or truncated:
            break

    return {
        'seed': seed,
        'return': total_return,
        'cost': total_cost,
        'true_cost': total_true_cost,
        'exposed_cost': total_exposed_cost,
        'steps': steps,
        'violated': int(total_true_cost > 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointGoal1-v0')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--out', default='results/random_clean.csv')
    parser.add_argument(
        '--corruption',
        choices=['none', 'lidar_blind', 'cost_under'],
        default='none',
    )
    parser.add_argument('--blind-dims', type=int, default=8)
    parser.add_argument('--cost-scale', type=float, default=0.25)
    args = parser.parse_args()

    env = safety_gymnasium.make(args.env)
    if args.corruption == 'lidar_blind':
        env = LidarBlindnessWrapper(env, blind_dims=args.blind_dims, fill_value=1.0)
    elif args.corruption == 'cost_under':
        env = CostUnderestimateWrapper(env, scale=args.cost_scale)

    rows = [run_episode(env, seed) for seed in trange(args.episodes, desc='episodes')]
    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=['seed', 'return', 'cost', 'true_cost', 'exposed_cost', 'steps', 'violated'],
    )
    df.to_csv(out_path, index=False)

    print('avg_return:', float(df['return'].mean()))
    print('avg_cost:', float(df['cost'].mean()))
    print('avg_true_cost:', float(df['true_cost'].mean()))
    print('violation_rate:', float(df['violated'].mean()))
    print('corruption:', args.corruption)
    print('csv:', out_path)


if __name__ == '__main__':
    main()

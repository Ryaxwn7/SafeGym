# PPO Auxiliary Reference

## Experiment Setup

- Date: 2026-06-13
- Environment: `SafetyPointGoal1-v0`
- Python env: `safegym`, Python 3.8.20
- PPO library: Stable-Baselines3 2.3.2, PyTorch 2.3.1 CPU
- Evaluation episodes: 20 per policy
- Episode length: 1000 steps
- Safety metric: true Safety-Gymnasium `cost`

## Policies

- `random_clean`: seeded random action baseline from the perturbation batch.
- `ppo_50k_eval`: PPO trained for 50k steps with shaped reward `reward - 1.0 * cost`.
- `ppo_reward_only_100k_eval`: PPO trained for 100k steps with original reward only.

## Summary

| Policy | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: |
| random_clean | 0.118627 | 35.30 | 0.20 |
| ppo_50k_eval | -0.770043 | 63.35 | 0.35 |
| ppo_reward_only_100k_eval | 19.736534 | 84.30 | 1.00 |

## Interpretation

The reward-only PPO policy is a stronger task policy than random: average return increases from `0.118627` to `19.736534`. However, it is not safety-aware and violates safety in every evaluation episode, raising average true cost to `84.30`.

The 50k cost-penalty PPO run did not outperform random. It is retained as a useful negative result: a short PPO run with a naive cost penalty is not enough to produce a reliable safe policy in this environment.

For the course project, PPO is auxiliary context rather than the main comparison target. The key takeaway is that reward optimization alone can increase true safety cost. The main project question is now whether SafeDreamer remains safe when its observations or safety-cost signals are attacked.

## Outputs

- Reward-only PPO model: `models/ppo_safety_point_goal_reward_only_100k.zip`
- Cost-penalty PPO model: `models/ppo_safety_point_goal_50k.zip`
- Reward-only PPO CSV: `results/2026-06-13-ppo-baseline/ppo_reward_only_100k_eval.csv`
- Cost-penalty PPO CSV: `results/2026-06-13-ppo-baseline/ppo_50k_eval.csv`
- Summary CSV: `results/2026-06-13-ppo-baseline/summary.csv`
- Summary figure: `figures/2026-06-13-ppo-baseline/summary_metrics.png`
- Reward-only PPO replay: `figures/replays/ppo_reward_only_100k_seed0.gif`

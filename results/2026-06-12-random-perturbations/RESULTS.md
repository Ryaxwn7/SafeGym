# Safety-Gymnasium Random Perturbation Results

## Experiment Setup

- Date: 2026-06-12
- Environment: `conda` env `safegym`, Python 3.8.20
- Task: `SafetyPointGoal1-v0`
- Policy: random action sampling with seeded `env.action_space`
- Episodes: 20 per condition
- Step API: `obs, reward, cost, terminated, truncated, info = env.step(action)`

## Conditions

- `clean`: original Safety-Gymnasium environment.
- `lidar_blind`: masks the last 8 observation dimensions with `1.0`.
- `cost_under`: exposes `cost * 0.25` while preserving `true_cost` in `info`.

## Summary

| Condition | Avg Return | Avg Cost | Avg True Cost | Avg Exposed Cost | Violation Rate | Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.118627 | 35.300 | 35.300 | 35.300 | 0.200 | 1000.0 |
| lidar_blind | 0.118627 | 35.300 | 35.300 | 35.300 | 0.200 | 1000.0 |
| cost_under | 0.118627 | 8.825 | 35.300 | 8.825 | 0.200 | 1000.0 |

## Interpretation

The `lidar_blind` condition matches `clean` because this baseline uses a random policy; the corrupted observation is not used for action selection. This wrapper is still useful for the next stage with learned policies or safety predictors.

The `cost_under` condition shows the intended safety-signal distortion: exposed average cost drops from `35.300` to `8.825`, while true average cost remains `35.300`. This demonstrates why later experiments must evaluate true environment cost rather than only the cost visible to a planner or safety head.

## Outputs

- Clean CSV: `results/2026-06-12-random-perturbations/random_clean.csv`
- Lidar-blind CSV: `results/2026-06-12-random-perturbations/random_lidar_blind.csv`
- Cost-under CSV: `results/2026-06-12-random-perturbations/random_cost_under.csv`
- Summary CSV: `results/2026-06-12-random-perturbations/summary.csv`
- Summary figure: `figures/2026-06-12-random-perturbations/summary_metrics.png`

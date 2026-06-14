# SafeDreamer Clean Smoke Evaluation

## Setup

- Date: 2026-06-13
- Environment: `SafetyPointGoal1-v0`
- SafeDreamer task: `safetygymcoor_SafetyPointGoal1-v0`
- Checkpoint: `external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt`
- Python env: `safedreamer-py38`
- JAX/JAXLIB: `0.3.25` / `0.3.25`, CPU backend
- Logger change: TensorBoard output is skipped when TensorFlow is not installed; JSONL logs are still written.

## Clean Baseline Result

SafeDreamer loaded the existing OSRP-Vector checkpoint and completed two clean evaluation episodes before the original TensorBoard logger failure. Those episodes were preserved in `scores.jsonl` and collected into `safedreamer_clean.csv`.

| Method | Episodes | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: | ---: |
| safedreamer_clean | 2 | 6.279988 | 0.00 | 0.00 |

## Auxiliary Context

| Method | Episodes | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: | ---: |
| random_clean | 20 | 0.118627 | 35.30 | 0.20 |
| ppo_reward_only_100k_eval | 20 | 19.736534 | 84.30 | 1.00 |
| safedreamer_clean | 2 | 6.279988 | 0.00 | 0.00 |

## Interpretation

The SafeDreamer checkpoint is the primary subject of the project. This clean run establishes that the OSRP-Vector checkpoint can be loaded and evaluated in `SafetyPointGoal1-v0`, and it gives the clean reference point for later attack experiments.

The random and PPO rows are auxiliary context only. They are not the main comparison target. The next required experiment is to evaluate the same SafeDreamer checkpoint under observation and cost-signal attacks such as `lidar_blind`, `hazard_blind`, and `cost_under`, then compare true cost and violation rate against this clean baseline.

This is not a final statistical conclusion: SafeDreamer was evaluated for only two episodes on CPU because each policy step is slow and the original eval path spends extra time generating prediction plots/videos. A stronger final result should run 10-20 episodes per attack condition or patch eval-only mode to disable report plotting.

## Outputs

- SafeDreamer CSV: `results/2026-06-13-safedreamer-eval/safedreamer_clean.csv`
- Combined summary: `results/2026-06-13-safedreamer-eval/summary.csv`
- Summary figure: `figures/2026-06-13-safedreamer-eval/summary_metrics.png`
- SafeDreamer logdir: `results/2026-06-13-safedreamer-eval/20260613-214859_name_safetygymcoor_SafetyPointGoal1-v0_0`
- Replay videos: `groundtruth_video_far_list_201.mp4`, `groundtruth_video_far_list_402.mp4`

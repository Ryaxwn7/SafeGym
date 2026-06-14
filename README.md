# SafeGym Reliability Experiments

本仓库是课程小组项目工作空间，用于研究 Safety-Gymnasium 中强化学习策略的安全可靠性。当前阶段聚焦 `SafetyPointGoal1-v0`：比较随机策略、PPO 策略和现有 SafeDreamer 世界模型策略在任务收益和真实安全代价上的差异。

## 项目目的

Safety-Gymnasium 的 `step()` 返回 6 个值：

```python
obs, reward, cost, terminated, truncated, info = env.step(action)
```

其中 `cost` 是真实安全代价，也是本项目的核心评价指标。项目目标不是只追求更高 reward，而是验证策略在提高任务表现时是否会增加安全违规，并进一步测试世界模型方法是否能改善安全性。

## 实现方案

实验统一使用 `SafetyPointGoal1-v0`，按低算力路线逐步推进：

1. 跑通 Safety-Gymnasium smoke test，确认环境、观测空间、动作空间和 cost 返回值正常。
2. 生成随机策略 baseline，并保存 CSV 和统计图。
3. 加入观测扰动和 cost 欺骗 wrapper，测试安全信号失真对评估的影响。
4. 训练 PPO 作为更强的 model-free 对比策略。
5. 引入 SafeDreamer 现有 checkpoint，做世界模型策略的初步安全性评估。

## 工作空间结构

- `scripts/`：实验、训练、评估、分析和 replay 脚本。
- `results/`：CSV、汇总表和结果说明。
- `figures/`：统计图和部分 replay 文件。
- `models/`：本地 PPO 模型权重，默认不提交 Git。
- `external/SafeDreamer/`：用于 checkpoint 评估的 SafeDreamer 源码。
- `safety-gymnasium-main/`：本地 Safety-Gymnasium 源码副本，体积较大，默认不提交 Git。
- `AGENTS.md`：面向代码协作 agent 的贡献指南。

## 环境安装

本项目使用两个独立 conda 环境。推荐在新机器上从仓库根目录执行以下命令。

### safegym：常规实验环境

用于 Safety-Gymnasium smoke test、随机 baseline、PPO 训练、结果分析和 replay 录制：

```bash
conda create -n safegym python=3.8 -y
conda activate safegym
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/safegym.txt
```

安装后检查：

```bash
python -c "import safety_gymnasium; print('OK')"
python scripts/check_safety_gym.py
```

### safedreamer-py38：SafeDreamer 环境

用于 SafeDreamer checkpoint 评估和后续从头训练。该代码依赖旧版 JAX，默认 requirements 使用 CPU 版 `jaxlib==0.3.25`：

```bash
conda create -n safedreamer-py38 python=3.8 -y
conda activate safedreamer-py38
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/safedreamer-py38.txt
```

如果在 NVIDIA GPU 上训练，可先按 SafeDreamer 原 README 将 CPU 版 `jaxlib` 替换为 CUDA 11 版本：

```bash
python -m pip uninstall -y jaxlib
python -m pip install jaxlib==0.3.25+cuda11.cudnn82 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

RTX 40 系显卡可以训练低维 `osrp_vector` 任务，但旧 JAX/CUDA 组合可能需要匹配驱动和 CUDA 兼容包。若 GPU 安装不稳定，先用 CPU 跑 checkpoint smoke test，再处理 GPU 训练环境。

当前仓库保留 `external/SafeDreamer/` 小型源码副本，但不提交下载的 checkpoint 和运行日志。checkpoint 需单独下载到 `external/checkpoints/`。

## 已实现内容

### Safety-Gymnasium Smoke Test

已实现：

- `scripts/check_safety_gym.py`
- `scripts/inspect_obs_space.py`

当前环境观测维度为 `(60,)`，动作空间为 `Box(-1.0, 1.0, (2,), float64)`。

运行：

```bash
conda run -n safegym python scripts/check_safety_gym.py
conda run -n safegym python scripts/inspect_obs_space.py
```

### 随机策略与扰动实验

已实现：

- `scripts/run_random_baseline.py`
- `scripts/obs_corruption_wrapper.py`
- `scripts/cost_shift_wrapper.py`
- `scripts/analyze_results.py`

20 episode 结果：

| 条件 | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: |
| clean | 0.1186 | 35.30 | 0.20 |
| lidar_blind | 0.1186 | 35.30 | 0.20 |
| cost_under | 0.1186 | 35.30 | 0.20 |

随机策略不使用观测，因此 `lidar_blind` 与 clean 一致；`cost_under` 用于验证暴露 cost 被低估时，真实 cost 仍可从 `info` 中保留。

### PPO 对比策略

已实现：

- `scripts/train_ppo_baseline.py`
- `scripts/record_ppo_replay.py`

20 episode 结果：

| 策略 | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: |
| random_clean | 0.1186 | 35.30 | 0.20 |
| ppo_50k_eval | -0.7700 | 63.35 | 0.35 |
| ppo_reward_only_100k_eval | 19.7365 | 84.30 | 1.00 |

reward-only PPO 的任务收益明显高于随机策略，但安全违规率达到 100%，说明只优化 reward 会牺牲安全性。

### SafeDreamer 初步评估

已实现：

- `scripts/run_safedreamer_eval.py`
- `scripts/collect_safedreamer_results.py`
- `external/SafeDreamer/SafeDreamer/train.py` 中加入了无 TensorFlow 时跳过 TensorBoard 输出的兼容补丁。

使用 checkpoint：

```text
Weidong-Huang/SafeDreamer:
safedreamer_osrp_vector/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt
```

初步结果：

| 方法 | Episodes | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: | ---: |
| random_clean | 20 | 0.1186 | 35.30 | 0.20 |
| ppo_reward_only_100k_eval | 20 | 19.7365 | 84.30 | 1.00 |
| safedreamer_clean | 2 | 6.2800 | 0.00 | 0.00 |

SafeDreamer checkpoint 已能加载并完成 CPU smoke evaluation。2 个 episode 中 true cost 为 0，但样本数还不足以作为最终结论。

## 常用命令

运行随机 baseline：

```bash
conda run -n safegym python scripts/run_random_baseline.py \
  --episodes 20 \
  --corruption none \
  --out results/random_clean.csv
```

训练并评估 PPO：

```bash
conda run -n safegym python scripts/train_ppo_baseline.py \
  --timesteps 100000 \
  --cost-penalty 0.0 \
  --eval-episodes 20 \
  --model-out models/ppo_safety_point_goal_reward_only_100k.zip \
  --eval-out results/2026-06-13-ppo-baseline/ppo_reward_only_100k_eval.csv
```

运行 SafeDreamer checkpoint 评估：

```bash
python scripts/run_safedreamer_eval.py --steps 1000
conda run -n safegym python scripts/collect_safedreamer_results.py
```

录制 replay：

```bash
conda run -n safegym python scripts/record_safety_gym_replay.py \
  --seed 2 \
  --steps 1000 \
  --frame-skip 5 \
  --out figures/replays/safety_point_goal_seed2_full.gif
```

## 关键输出

- 随机扰动实验：`results/2026-06-12-random-perturbations/`
- PPO 实验：`results/2026-06-13-ppo-baseline/`
- SafeDreamer 实验：`results/2026-06-13-safedreamer-eval/`
- 汇总图：
  - `figures/2026-06-12-random-perturbations/summary_metrics.png`
  - `figures/2026-06-13-ppo-baseline/summary_metrics.png`
  - `figures/2026-06-13-safedreamer-eval/summary_metrics.png`

## 未实现内容

- SafeDreamer 在 `hazard_blind`、`cost_under` 等扰动条件下的系统评估。
- 10-20 episode 的 SafeDreamer 统计对比。
- SafeDreamer 重新训练或微调。
- 学习式 cost predictor 的备用实验。
- 最终课程展示 slides 和完整实验报告。

## Git 与共享说明

当前工作空间已整理为单一根 Git 仓库，子目录中的嵌套 `.git` 已移除。`.gitignore` 默认忽略大文件和生成文件，包括 `main.zip`、checkpoint、PPO 模型、缓存、TensorBoard 文件、视频和 replay 媒体。

首次提交可使用：

```bash
git add .
git commit -m "Initial SafeGym reliability experiments"
```

如需上传 GitHub，再添加远程仓库：

```bash
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```

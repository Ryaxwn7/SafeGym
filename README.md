# SafeDreamer Attack Robustness Experiments

本仓库是课程小组项目工作空间，用于评估 SafeDreamer 世界模型策略在 Safety-Gymnasium 安全任务中的攻击鲁棒性。核心问题是：当机器人策略依赖世界模型和安全代价预测时，观测被遮蔽、危险信息缺失或 cost 信号被低估，会不会导致策略低估真实风险。

当前环境聚焦 `SafetyPointGoal1-v0`。Safety-Gymnasium 提供真实仿真环境和 ground-truth safety cost，SafeDreamer 提供 world-model-based policy/planning 方法。

## 研究目标

Safety-Gymnasium 的 step API 为：

```python
obs, reward, cost, terminated, truncated, info = env.step(action)
```

其中 `cost` 是真实安全代价，也是本项目主指标。项目重点不是比较 PPO 和 SafeDreamer 谁性能更好，而是测试 SafeDreamer 在不同攻击/扰动条件下的安全性能是否退化。

主要评价指标：

- `true_cost`：真实环境 cost，作为安全标签。
- `exposed_cost`：策略或 wrapper 暴露出来的 cost，可能被攻击低估。
- `violation_rate`：episode 中是否出现安全违规。
- `return`：任务收益，用于观察安全和任务表现的权衡。
- `attack_delta`：攻击条件相对 clean 条件的 true cost / violation rate 变化。

## 实验流程

整体流程按以下顺序推进：

1. **环境验证**：跑通 Safety-Gymnasium，确认 `SafetyPointGoal1-v0` 的 observation、action 和 6 返回值接口正常。
2. **攻击 wrapper 构建**：实现观测攻击和 cost 信号攻击，确保真实 cost 仍可用于离线评估。
3. **随机策略 sanity check**：用随机策略验证 wrapper、CSV、分析脚本和图表生成流程。
4. **SafeDreamer clean baseline**：加载现有 OSRP-Vector checkpoint，在 clean 环境下得到 SafeDreamer 的基准安全表现。
5. **SafeDreamer attack evaluation**：在相同 checkpoint 下运行 `lidar_blind`、`hazard_blind`、`cost_under` 等攻击条件，比较 true cost 和 violation rate 是否恶化。
6. **辅助参照**：PPO 只用于说明 reward-only 策略可能不安全，不作为主实验结论。

## 攻击条件设计

| 条件 | 目的 | 当前状态 |
| --- | --- | --- |
| `clean` | 无攻击基准 | 已用于随机策略和 SafeDreamer smoke eval |
| `lidar_blind` | 遮蔽部分 lidar/观测维度，模拟危险感知缺失 | 已在随机策略 wrapper 中验证 |
| `cost_under` | 将暴露 cost 缩小为 `0.25 * cost`，模拟安全反馈低估 | 已在随机策略 wrapper 中验证 |
| `hazard_blind` | 精确遮蔽 hazard 相关观测或坐标 | 待实现 |
| SafeDreamer attacked eval | 将上述攻击接入 SafeDreamer 环境封装 | 待实现 |

## 可实现的 SafeDreamer 攻击方式

本项目只研究仿真中的输入、反馈和执行链路扰动，不涉及真实机器人或系统入侵。推荐按以下优先级实现：

| 攻击方式 | 作用对象 | 实现难度 | 课程项目价值 |
| --- | --- | --- | --- |
| `hazard_blind` | hazard 相关观测/坐标 | 中 | 直接测试危险感知缺失，是最贴合安全主题的主攻击 |
| `lidar_blind` | lidar 或后若干维 observation | 低 | 已有随机策略 wrapper，可快速接入 SafeDreamer |
| `obs_noise` | observation 向量 | 低 | 可设置噪声强度，画出攻击强度曲线 |
| `obs_delay` | observation 时间序列 | 中 | 模拟传感器延迟，适合移动机器人安全场景 |
| `cost_under` | 暴露给策略/日志的 cost | 低 | 测试安全反馈低估；对重新训练/微调实验更关键 |
| `action_noise` | 执行动作 | 低 | 测试策略对执行误差的鲁棒性 |
| `action_delay` | 执行动作时间序列 | 中 | 模拟控制延迟，适合作为扩展实验 |

优先实现的主实验组合：

| 条件 | 说明 |
| --- | --- |
| `safedreamer_clean` | 无攻击基准 |
| `safedreamer_hazard_blind` | 遮蔽 hazard 信息，观察 true cost 是否上升 |
| `safedreamer_lidar_blind` | 遮蔽部分观测，验证感知缺失影响 |
| `safedreamer_obs_noise_0.1` | 添加中等强度观测噪声 |
| `safedreamer_cost_under` | 暴露 cost 被低估，但评估仍使用真实 cost |

不建议作为本课程主线的攻击：直接篡改 SafeDreamer 网络权重、对 JAX checkpoint 做参数级攻击、图像对抗样本、真实系统攻击。这些方向实现成本高，且会偏离当前 `osrp_vector` 低维观测路线。

实现入口：

- Safety-Gymnasium/random sanity wrapper：`scripts/obs_corruption_wrapper.py`、`scripts/cost_shift_wrapper.py`
- SafeDreamer 环境封装：`external/SafeDreamer/SafeDreamer/embodied/envs/safetygymcoor.py`
- 结果分析：`scripts/analyze_results.py`、`scripts/collect_safedreamer_results.py`

## 工作空间结构

- `scripts/`：实验、训练、评估、分析和 replay 脚本。
- `results/`：CSV、汇总表和结果说明。
- `figures/`：统计图和部分 replay 文件。
- `requirements/`：两个 conda 环境对应的 pip requirements。
- `models/`：本地 PPO 模型权重，默认不提交 Git。
- `external/SafeDreamer/`：用于 checkpoint 评估和后续训练的 SafeDreamer 源码。
- `safety-gymnasium-main/`：本地 Safety-Gymnasium 源码副本，体积较大，默认不提交 Git。
- `AGENTS.md`：面向代码协作 agent 的贡献指南。

## 环境安装

本项目使用两个独立 conda 环境。推荐在新机器上从仓库根目录执行以下命令。

### safegym：常规实验环境

用于 Safety-Gymnasium smoke test、攻击 wrapper 验证、随机 sanity baseline、PPO 辅助参照、结果分析和 replay 录制：

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

如果在 NVIDIA GPU 上训练，可按 SafeDreamer 原 README 将 CPU 版 `jaxlib` 替换为 CUDA 11 版本：

```bash
python -m pip uninstall -y jaxlib
python -m pip install jaxlib==0.3.25+cuda11.cudnn82 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

RTX 40 系显卡可以训练低维 `osrp_vector` 任务，但旧 JAX/CUDA 组合可能需要匹配驱动和 CUDA 兼容包。若 GPU 安装不稳定，先用 CPU 跑 checkpoint smoke test，再处理 GPU 训练环境。

checkpoint 需单独下载到 `external/checkpoints/`，不会提交到 Git。

## 已实现内容

### 1. Safety-Gymnasium 环境验证

已实现：

- `scripts/check_safety_gym.py`
- `scripts/inspect_obs_space.py`

当前环境观测维度为 `(60,)`，动作空间为 `Box(-1.0, 1.0, (2,), float64)`。

运行：

```bash
conda run -n safegym python scripts/check_safety_gym.py
conda run -n safegym python scripts/inspect_obs_space.py
```

### 2. 攻击 Wrapper 与随机策略验证

已实现：

- `scripts/run_random_baseline.py`
- `scripts/obs_corruption_wrapper.py`
- `scripts/cost_shift_wrapper.py`
- `scripts/analyze_results.py`

20 episode sanity check：

| 条件 | Avg Return | Avg True Cost | Avg Exposed Cost | Violation Rate |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.1186 | 35.30 | 35.30 | 0.20 |
| lidar_blind | 0.1186 | 35.30 | 35.30 | 0.20 |
| cost_under | 0.1186 | 35.30 | 8.825 | 0.20 |

随机策略不使用观测，因此 `lidar_blind` 与 clean 一致。这部分的意义是验证攻击和统计链路。`cost_under` 已能制造暴露 cost 与真实 cost 的偏差。

### 3. SafeDreamer Clean Baseline

已实现：

- `scripts/run_safedreamer_eval.py`
- `scripts/collect_safedreamer_results.py`
- `external/SafeDreamer/SafeDreamer/train.py` 中加入了无 TensorFlow 时跳过 TensorBoard 输出的兼容补丁。

使用 checkpoint：

```text
Weidong-Huang/SafeDreamer:
safedreamer_osrp_vector/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt
```

clean smoke result：

| 条件 | Episodes | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: | ---: |
| safedreamer_clean | 2 | 6.2800 | 0.00 | 0.00 |

该结果只说明 SafeDreamer checkpoint 能在当前环境加载并完成 clean evaluation。样本数少，且尚未接入攻击条件，因此不能作为最终安全结论。

### 4. PPO 辅助参照

已实现：

- `scripts/train_ppo_baseline.py`
- `scripts/record_ppo_replay.py`

20 episode 结果：

| 策略 | Avg Return | Avg True Cost | Violation Rate |
| --- | ---: | ---: | ---: |
| random_clean | 0.1186 | 35.30 | 0.20 |
| ppo_50k_eval | -0.7700 | 63.35 | 0.35 |
| ppo_reward_only_100k_eval | 19.7365 | 84.30 | 1.00 |

PPO 结果作为辅助背景：reward-only 策略可以获得更高任务收益，但安全违规显著增加。主实验仍是 SafeDreamer 在攻击条件下的 true cost 变化。

## 常用命令

运行随机攻击 sanity check：

```bash
conda run -n safegym python scripts/run_random_baseline.py \
  --episodes 20 \
  --corruption cost_under \
  --out results/random_cost_under.csv
```

运行 SafeDreamer clean checkpoint 评估：

```bash
python scripts/run_safedreamer_eval.py --steps 1000
conda run -n safegym python scripts/collect_safedreamer_results.py
```

运行完整 SafeDreamer 攻击测试套件：

```bash
python scripts/run_safedreamer_attack_suite.py --episodes 1 --continue-on-error
```

脚本会按条件顺序运行，实时打印 `[时间][条件]` 进度，并把每个条件的完整日志保存到：

```text
results/2026-06-14-safedreamer-attacks/<condition>/run.log
```

### Episode 数量建议

SafeDreamer CPU eval 较慢，不建议一开始就跑大量 episode。推荐分三档：

| 用途 | 每个条件 episode 数 | 说明 |
| --- | ---: | --- |
| smoke test | 1-2 | 确认环境、攻击 wrapper、日志和 CSV 能跑通 |
| 课程报告初步结果 | 5 | 能观察 clean vs attack 趋势，适合当前阶段 |
| 最终对比结果 | 10 | 更适合放入最终表格；若时间充足再扩到 20 |

推荐运行顺序：

```bash
# 1. 先跑通全部条件
python scripts/run_safedreamer_attack_suite.py --episodes 1 --continue-on-error

# 2. 生成课程项目主结果
python scripts/run_safedreamer_attack_suite.py --episodes 5 --force --continue-on-error

# 3. 时间充足时生成最终对比
python scripts/run_safedreamer_attack_suite.py --episodes 10 --force --continue-on-error
```

已有 CSV 默认跳过，适合中断后继续跑。常用选项：

```bash
python scripts/run_safedreamer_attack_suite.py --list
python scripts/run_safedreamer_attack_suite.py --only safedreamer_cost_under
python scripts/run_safedreamer_attack_suite.py --force --only safedreamer_clean
python scripts/run_safedreamer_attack_suite.py --episodes 3 --force
```

当前 `SafetyPointGoal1-v0` 在 SafeDreamer `safetygymcoor` 配置中 `repeat=5`，一个完整 episode 约为 200 个 policy step。脚本默认用 `--episode-length 200`，并自动计算 `steps = episodes * episode_length + step_margin`，避免只跑半个 episode 后没有 CSV。

套件输出：

```text
results/2026-06-14-safedreamer-attacks/all_episodes.csv
results/2026-06-14-safedreamer-attacks/summary.csv
results/2026-06-14-safedreamer-attacks/failures.csv  # only if a condition fails
```

训练并评估 PPO 辅助参照：

```bash
conda run -n safegym python scripts/train_ppo_baseline.py \
  --timesteps 100000 \
  --cost-penalty 0.0 \
  --eval-episodes 20 \
  --model-out models/ppo_safety_point_goal_reward_only_100k.zip \
  --eval-out results/2026-06-13-ppo-baseline/ppo_reward_only_100k_eval.csv
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

- 攻击 wrapper sanity check：`results/2026-06-12-random-perturbations/`
- SafeDreamer clean smoke eval：`results/2026-06-13-safedreamer-eval/`
- PPO 辅助参照：`results/2026-06-13-ppo-baseline/`
- 汇总图：
  - `figures/2026-06-12-random-perturbations/summary_metrics.png`
  - `figures/2026-06-13-safedreamer-eval/summary_metrics.png`
  - `figures/2026-06-13-ppo-baseline/summary_metrics.png`

## 下一步

- 将 `lidar_blind`、`cost_under` 攻击接入 SafeDreamer 的 `safetygymcoor` 环境封装。
- 新增 `hazard_blind`，只遮蔽 hazard 相关观测或坐标。
- 新增 `obs_noise` 和 `action_noise`，支持设置攻击强度。
- 对 `safedreamer_clean`、`safedreamer_hazard_blind`、`safedreamer_lidar_blind`、`safedreamer_obs_noise_0.1`、`safedreamer_cost_under` 运行 10-20 episode。
- 统计 attack delta：`true_cost`、`violation_rate`、`return` 相对 clean 的变化。
- 如时间允许，在受攻击数据或新约束下重新训练/微调 SafeDreamer。
- 准备最终课程报告和 slides，强调 Safety-Gymnasium 是真实安全环境，SafeDreamer 是被测世界模型策略。

## Git 与共享说明

当前工作空间已整理为单一根 Git 仓库，子目录中的嵌套 `.git` 已移除。`.gitignore` 默认忽略大文件和生成文件，包括 `main.zip`、checkpoint、PPO 模型、缓存、TensorBoard 文件、视频和 replay 媒体。

首次提交可使用：

```bash
git add .
git commit -m "Initial SafeDreamer attack robustness experiments"
```

如需上传 GitHub，再添加远程仓库：

```bash
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```

# SafeDreamer Safety-Gymnasium Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成一个低算力可复现的可信大模型课程大作业：评估危险/近失误扰动下机器人安全世界模型的可靠性。

**Architecture:** Safety-Gymnasium 只作为安全仿真环境和真实 `cost` 来源；SafeDreamer 作为 world-model-based safe RL 方法；实验比较 clean、观测污染、cost 低估三种条件下的 reward、true cost、cost violation rate 和轨迹行为变化。若 SafeDreamer 环境安装失败，降级为 Safety-Gymnasium + 轻量 learned safety predictor 的机制复现实验，但汇报中明确说明降级边界。

**Tech Stack:** Python 3.8 conda/venv, Safety-Gymnasium, SafeDreamer, JAX 0.3.25, NumPy, Pandas, Matplotlib, PowerPoint/Markdown.

---

## Final Topic

中文题目：

**基于 SafeDreamer 与 Safety-Gymnasium 的机器人安全世界模型可靠性评估**

备用更短题目：

**近失误扰动下安全世界模型对机器人策略的误导风险评估**

核心表述：

- Safety-Gymnasium 是 benchmark，不提供 learned world model。
- SafeDreamer 是 world-model-based safe RL 方法，负责把 world model 和安全规划结合起来。
- 本项目不研究真实攻击操作，而研究安全评估器/世界模型在 near-miss、危险观测扰动、cost 低估时是否会误导策略。

## Deliverables

- [ ] `safe_wm_reliability/README.md`：项目说明、环境安装、实验命令、结果解释。
- [ ] `safe_wm_reliability/scripts/check_safety_gym.py`：Safety-Gymnasium smoke test。
- [ ] `safe_wm_reliability/scripts/run_random_baseline.py`：无学习随机策略基线，确认 reward/cost 指标采集正确。
- [ ] `safe_wm_reliability/scripts/obs_corruption_wrapper.py`：观测扰动 wrapper，用于模拟危险感知缺失。
- [ ] `safe_wm_reliability/scripts/cost_shift_wrapper.py`：cost 低估 wrapper，用于模拟安全信号污染。
- [ ] `safe_wm_reliability/scripts/run_safedreamer_eval.ps1`：SafeDreamer checkpoint 评估命令集合。
- [ ] `safe_wm_reliability/scripts/analyze_results.py`：聚合结果，输出表格和图。
- [ ] `safe_wm_reliability/results/`：CSV、图表、关键轨迹截图。
- [ ] `safe_wm_reliability/report_outline.md`：汇报稿结构。
- [ ] `25113070049-张逸樵-SafeWM-Reliability.pptx`：最终课程汇报 PPT。

## Phase 0: Scope Lock

- [ ] **Step 0.1: 固定研究问题**

  写入 `safe_wm_reliability/README.md`：

  ```markdown
  # Safe World Model Reliability

  本项目评估安全世界模型在机器人安全 RL 任务中的可靠性。Safety-Gymnasium 提供真实仿真环境和 cost；SafeDreamer 提供 world-model-based safe RL 策略/规划框架。我们比较 clean、观测扰动、cost 低估三种条件下的真实安全代价变化，分析 world model 或安全信号错误如何使策略高估安全性。
  ```

- [ ] **Step 0.2: 固定最低成功标准**

  最低可交付必须包含：

  - clean 条件下能跑通 `SafetyPointGoal1-v0` 或 `SafetyPointCircle0-v0`。
  - 能采集每个 episode 的 `return`、`cost`、`cost_violation_rate`。
  - 至少一种扰动条件与 clean 做对比。
  - PPT 中明确说明 Safety-Gymnasium 不提供 WM，SafeDreamer 才是 WM 方法。

- [ ] **Step 0.3: 固定增强成功标准**

  增强可交付包含：

  - SafeDreamer OSRP-Vector checkpoint 跑通 CPU 或 GPU eval。
  - clean / corrupted 的真实 cost 有可视化差异。
  - 至少一张轨迹或环境截图解释 near-miss 场景。

## Phase 1: Literature and Framing

- [ ] **Step 1.1: 阅读并摘录 SafeDreamer 关键信息**

  来源：`https://github.com/PKU-Alignment/SafeDreamer`

  摘录点：

  - SafeDreamer 是 “Safe Reinforcement Learning with World Models”。
  - 它区分 reward 与 cost，并在 world model 内做 safety-reward planning。
  - 它在 Safety-Gymnasium 上评估。
  - 它提供 OSRP-Vector 小 checkpoint，约 26.6MB，适合低算力验证。

- [ ] **Step 1.2: 阅读并摘录 Safety-Gymnasium 关键信息**

  来源：`https://safety-gymnasium.readthedocs.io/en/latest/introduction/basic_usage.html`

  摘录点：

  - Safety-Gymnasium 是 Safe RL benchmark。
  - `env.step(action)` 返回 `obs, reward, cost, terminated, truncated, info`。
  - `cost` 是本项目的真实安全标签。
  - `SafetyPointGoal1-v0` / `SafetyPointCircle0-v0` 是首选低算力任务。

- [ ] **Step 1.3: 连接 World Model / WMP / World-VLA-Loop**

  写入 `report_outline.md`：

  ```markdown
  ## Motivation

  机器人策略越来越依赖 world model 预测动作后果。WMP 用 world model 改善腿式机器人视觉运动感知；World-VLA-Loop 强调 near-success 数据对 reward alignment 的重要性。我们的项目不复现大型视频世界模型，而是在 SafeDreamer + Safety-Gymnasium 上做低算力机制复现：当危险观测或安全代价被污染时，world-model-based safe RL 是否会低估真实风险。
  ```

## Phase 2: Environment Setup

- [ ] **Step 2.1: 创建独立环境**

  PowerShell:

  ```powershell
  conda create -n safe-wm-py38 python=3.8 -y
  conda activate safe-wm-py38
  python -m pip install --upgrade pip
  ```

  Expected:

  ```text
  Python 3.8.x
  ```

- [ ] **Step 2.2: 安装 Safety-Gymnasium**

  PowerShell:

  ```powershell
  pip install safety-gymnasium
  ```

  Expected:

  ```text
  Successfully installed safety-gymnasium
  ```

- [ ] **Step 2.3: 创建 Safety-Gymnasium smoke test**

  Create: `safe_wm_reliability/scripts/check_safety_gym.py`

  ```python
  import safety_gymnasium

  env = safety_gymnasium.make("SafetyPointGoal1-v0")
  obs, info = env.reset(seed=0)
  total_reward = 0.0
  total_cost = 0.0

  for _ in range(100):
      action = env.action_space.sample()
      obs, reward, cost, terminated, truncated, info = env.step(action)
      total_reward += float(reward)
      total_cost += float(cost)
      if terminated or truncated:
          obs, info = env.reset()

  env.close()
  print({"total_reward": total_reward, "total_cost": total_cost})
  ```

- [ ] **Step 2.4: 运行 smoke test**

  PowerShell:

  ```powershell
  python safe_wm_reliability/scripts/check_safety_gym.py
  ```

  Expected:

  ```text
  {'total_reward': ..., 'total_cost': ...}
  ```

## Phase 3: Baseline Metric Collection

- [ ] **Step 3.1: 创建随机策略基线脚本**

  Create: `safe_wm_reliability/scripts/run_random_baseline.py`

  ```python
  import argparse
  import csv
  from pathlib import Path

  import safety_gymnasium


  def run_episode(env, seed):
      obs, info = env.reset(seed=seed)
      ep_return = 0.0
      ep_cost = 0.0
      steps = 0
      while True:
          action = env.action_space.sample()
          obs, reward, cost, terminated, truncated, info = env.step(action)
          ep_return += float(reward)
          ep_cost += float(cost)
          steps += 1
          if terminated or truncated:
              return {
                  "seed": seed,
                  "return": ep_return,
                  "cost": ep_cost,
                  "steps": steps,
                  "violated": int(ep_cost > 0),
              }


  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--env", default="SafetyPointGoal1-v0")
      parser.add_argument("--episodes", type=int, default=20)
      parser.add_argument("--out", default="safe_wm_reliability/results/random_clean.csv")
      args = parser.parse_args()

      out_path = Path(args.out)
      out_path.parent.mkdir(parents=True, exist_ok=True)

      env = safety_gymnasium.make(args.env)
      rows = [run_episode(env, seed) for seed in range(args.episodes)]
      env.close()

      with out_path.open("w", newline="", encoding="utf-8") as f:
          writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
          writer.writeheader()
          writer.writerows(rows)

      avg_return = sum(row["return"] for row in rows) / len(rows)
      avg_cost = sum(row["cost"] for row in rows) / len(rows)
      violation_rate = sum(row["violated"] for row in rows) / len(rows)
      print({"avg_return": avg_return, "avg_cost": avg_cost, "violation_rate": violation_rate})


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3.2: 运行随机策略 clean 基线**

  PowerShell:

  ```powershell
  python safe_wm_reliability/scripts/run_random_baseline.py --episodes 20 --out safe_wm_reliability/results/random_clean.csv
  ```

  Expected:

  ```text
  {'avg_return': ..., 'avg_cost': ..., 'violation_rate': ...}
  ```

## Phase 4: Perturbation Wrappers

- [ ] **Step 4.1: 创建观测污染 wrapper**

  Create: `safe_wm_reliability/scripts/obs_corruption_wrapper.py`

  ```python
  import gymnasium as gym
  import numpy as np


  class LidarBlindnessWrapper(gym.ObservationWrapper):
      """Mask the last lidar-like observation dimensions to simulate missed danger perception."""

      def __init__(self, env, blind_dims=8, fill_value=1.0):
          super().__init__(env)
          self.blind_dims = int(blind_dims)
          self.fill_value = float(fill_value)

      def observation(self, observation):
          obs = np.array(observation, copy=True)
          if obs.ndim == 1 and obs.shape[0] >= self.blind_dims:
              obs[-self.blind_dims :] = self.fill_value
          return obs
  ```

- [ ] **Step 4.2: 创建 cost 低估 wrapper**

  Create: `safe_wm_reliability/scripts/cost_shift_wrapper.py`

  ```python
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
          info["true_cost"] = true_cost
          info["exposed_cost"] = exposed_cost
          return obs, reward, exposed_cost, terminated, truncated, info
  ```

- [ ] **Step 4.3: 修改基线脚本支持扰动模式**

  Modify: `safe_wm_reliability/scripts/run_random_baseline.py`

  Add imports:

  ```python
  from obs_corruption_wrapper import LidarBlindnessWrapper
  from cost_shift_wrapper import CostUnderestimateWrapper
  ```

  Add arguments:

  ```python
  parser.add_argument("--corruption", choices=["none", "lidar_blind", "cost_under"], default="none")
  ```

  Wrap environment after creation:

  ```python
  env = safety_gymnasium.make(args.env)
  if args.corruption == "lidar_blind":
      env = LidarBlindnessWrapper(env, blind_dims=8, fill_value=1.0)
  elif args.corruption == "cost_under":
      env = CostUnderestimateWrapper(env, scale=0.25)
  ```

- [ ] **Step 4.4: 运行扰动基线**

  PowerShell:

  ```powershell
  python safe_wm_reliability/scripts/run_random_baseline.py --episodes 20 --corruption lidar_blind --out safe_wm_reliability/results/random_lidar_blind.csv
  python safe_wm_reliability/scripts/run_random_baseline.py --episodes 20 --corruption cost_under --out safe_wm_reliability/results/random_cost_under.csv
  ```

  Expected:

  ```text
  CSV files exist under safe_wm_reliability/results/
  ```

## Phase 5: SafeDreamer Checkpoint Evaluation

- [ ] **Step 5.1: 克隆 SafeDreamer**

  PowerShell:

  ```powershell
  git clone https://github.com/PKU-Alignment/SafeDreamer.git external/SafeDreamer
  ```

  Expected:

  ```text
  external/SafeDreamer/SafeDreamer/train.py exists
  ```

- [ ] **Step 5.2: 安装 SafeDreamer CPU 依赖**

  PowerShell:

  ```powershell
  cd external/SafeDreamer
  pip install jax==0.3.25
  pip install jaxlib==0.3.25
  pip install jax-jumpy==1.0.0
  pip install -r requirements.txt
  cd ../..
  ```

  Expected:

  ```text
  Successfully installed jax jaxlib jax-jumpy
  ```

- [ ] **Step 5.3: 下载 OSRP-Vector 小 checkpoint**

  PowerShell:

  ```powershell
  pip install huggingface_hub
  huggingface-cli download Weidong-Huang/SafeDreamer 20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt --subfolder safedreamer_osrp_vector --local-dir external/checkpoints/safedreamer_osrp_vector
  ```

  Expected:

  ```text
  external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt
  ```

- [ ] **Step 5.4: 创建 SafeDreamer eval 命令脚本**

  Create: `safe_wm_reliability/scripts/run_safedreamer_eval.ps1`

  ```powershell
  $Checkpoint = "external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt"
  python external/SafeDreamer/SafeDreamer/train.py `
    --configs osrp_vector `
    --method osrp `
    --run.script eval_only `
    --run.from_checkpoint $Checkpoint `
    --task safetygymcoor_SafetyPointGoal1-v0 `
    --jax.platform cpu `
    --run.steps 2000
  ```

- [ ] **Step 5.5: 运行 SafeDreamer eval**

  PowerShell:

  ```powershell
  powershell -ExecutionPolicy Bypass -File safe_wm_reliability/scripts/run_safedreamer_eval.ps1
  ```

  Expected:

  ```text
  SafeDreamer eval starts without import errors; logs include reward/cost related metrics.
  ```

## Phase 6: Result Analysis

- [ ] **Step 6.1: 创建结果分析脚本**

  Create: `safe_wm_reliability/scripts/analyze_results.py`

  ```python
  import argparse
  from pathlib import Path

  import matplotlib.pyplot as plt
  import pandas as pd


  def summarize(path):
      df = pd.read_csv(path)
      return {
          "file": Path(path).name,
          "episodes": len(df),
          "avg_return": df["return"].mean(),
          "avg_cost": df["cost"].mean(),
          "violation_rate": df["violated"].mean(),
      }


  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--inputs", nargs="+", required=True)
      parser.add_argument("--outdir", default="safe_wm_reliability/results")
      args = parser.parse_args()

      outdir = Path(args.outdir)
      rows = [summarize(path) for path in args.inputs]
      summary = pd.DataFrame(rows)
      summary_path = outdir / "summary.csv"
      summary.to_csv(summary_path, index=False)
      print(summary)

      fig, ax = plt.subplots(figsize=(7, 4))
      summary.plot(x="file", y=["avg_return", "avg_cost", "violation_rate"], kind="bar", ax=ax)
      ax.set_title("Safety metrics under clean and corrupted conditions")
      ax.set_ylabel("Metric value")
      ax.tick_params(axis="x", rotation=30)
      fig.tight_layout()
      fig.savefig(outdir / "summary_metrics.png", dpi=200)


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 6.2: 运行结果分析**

  PowerShell:

  ```powershell
  pip install pandas matplotlib
  python safe_wm_reliability/scripts/analyze_results.py --inputs safe_wm_reliability/results/random_clean.csv safe_wm_reliability/results/random_lidar_blind.csv safe_wm_reliability/results/random_cost_under.csv
  ```

  Expected:

  ```text
  safe_wm_reliability/results/summary.csv
  safe_wm_reliability/results/summary_metrics.png
  ```

## Phase 7: Report and Slides

- [ ] **Step 7.1: 写汇报结构**

  Create: `safe_wm_reliability/report_outline.md`

  ```markdown
  # 汇报结构

  1. 课程背景：可信大模型与具身智能安全
  2. 研究问题：world model 如果低估 near-miss 风险会怎样
  3. 方法边界：Safety-Gymnasium 是环境，SafeDreamer 是 WM 方法
  4. 实验设计：clean / lidar_blind / cost_under
  5. 指标：return、true cost、violation rate、轨迹行为
  6. 实验结果：表格和柱状图
  7. 讨论：对 WMP、World-VLA-Loop、机器人 RL 的启发
  8. 局限：非视频 WM、非真实腿式机器人、低算力小规模验证
  9. 结论：near-miss 与安全 cost 对 world-model-based RL 的可信性至关重要
  ```

- [ ] **Step 7.2: PPT 页数规划**

  PPT 建议 12 页：

  1. 标题页
  2. 课程关联与问题动机
  3. World Model 在机器人 RL 中的作用
  4. 为什么 WMP / World-VLA-Loop 难以低算力完整复现
  5. 本项目替代方案：Safety-Gymnasium + SafeDreamer
  6. 实验环境与任务
  7. 扰动设计：危险观测缺失与 cost 低估
  8. 指标定义
  9. Clean baseline 结果
  10. Corrupted 条件对比结果
  11. 可信具身智能讨论
  12. 总结与 Q&A

- [ ] **Step 7.3: 关键图表清单**

  PPT 至少包含：

  - 一张 Safety-Gymnasium 环境截图。
  - 一张 SafeDreamer 框架图，引用原论文或仓库图。
  - 一张 clean vs corrupted 指标柱状图。
  - 一张方法边界图：`Safety-Gymnasium -> obs/reward/cost -> SafeDreamer world model/planner -> policy -> true cost evaluation`。

## Phase 8: Risk Control and Fallbacks

- [ ] **Risk 8.1: SafeDreamer JAX 环境装不上**

  Fallback:

  - 继续保留 Safety-Gymnasium 结果。
  - 加一个轻量 `cost_predictor`，用 clean rollout 训练 `obs + action -> cost` 分类器。
  - 在污染数据上训练另一个 predictor，对比真实 cost 的 false negative rate。
  - 汇报标题降级为：**安全感知污染下机器人 RL 风险预测可靠性评估**，副标题说明这是 world-model safety head 的机制复现。

- [ ] **Risk 8.2: 运行时间不足**

  Fallback:

  - episodes 从 20 降到 5。
  - 只保留 `SafetyPointGoal1-v0`。
  - 只做 `lidar_blind` 一种扰动。
  - PPT 中把 SafeDreamer 作为方法分析，不强称完整复现。

- [ ] **Risk 8.3: 扰动差异不明显**

  Fallback:

  - 增大 `blind_dims` 到 12。
  - 把 `cost_under` 的 `scale` 从 0.25 降到 0.0。
  - 增加 episode 数到 50。
  - 用 violation rate 而不是 avg cost 作为主指标。

## Phase 9: Final Acceptance Checklist

- [ ] 能一句话解释：Safety-Gymnasium 不是 WM，SafeDreamer 才是 WM-based safe RL。
- [ ] 至少有一个脚本能稳定跑通 Safety-Gymnasium。
- [ ] 至少有 clean 和一种 corrupted 的 CSV 结果。
- [ ] 至少有一张结果图。
- [ ] PPT 没有把攻击写成可操作伤害流程，只讨论安全评估和防御启发。
- [ ] 结论能回扣课程主题：可信大模型/具身智能系统必须关注 near-miss 数据、reward/cost alignment、真实安全代价。

## Suggested Division of Labor

- 组员 A：环境安装、Safety-Gymnasium smoke test、随机基线。
- 组员 B：SafeDreamer checkpoint eval、记录安装问题和日志。
- 组员 C：扰动 wrapper、结果分析脚本、图表。
- 组员 D：论文动机、PPT、讲稿整合。

## Self-Review

- Spec coverage: 已覆盖选题边界、实验环境、WM 关系、低算力复现、汇报材料。
- Placeholder scan: 无 `TBD`、无“之后补充”式占位。
- Type consistency: `return/cost/violated` 字段在采集和分析脚本中一致。
- Known risk: SafeDreamer 依赖旧版 JAX，可能在 Windows 上安装受阻；计划中已给出 Safety-Gymnasium + predictor 降级路径。

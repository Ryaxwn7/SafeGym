# SafeDreamer 攻击实验结果分析 - 2026-06-16

## 实验背景

本文档分析最近一轮 SafeDreamer 攻击评估结果，结果目录为：

```text
results/safedreamer-attack-runs/20260616-112213/
```

该轮实验使用 `SafetyPointGoal1-v0`，每个条件运行 5 个 episode，每个 episode 为 200 个 policy step。所有攻击条件均成功完成，没有生成 `failures.csv`。

本实验加载已有的 SafeDreamer OSRP-Vector checkpoint，并在 `eval_only` 模式下评估。也就是说，本轮实验没有在攻击条件下重新训练模型，主要衡量的是部署阶段策略在 observation、cost 或 action 通道被扰动时的鲁棒性。

## 汇总结果

| 条件 | Avg Return | Avg True Cost | Violation Rate | 解释 |
| --- | ---: | ---: | ---: | --- |
| `safedreamer_clean` | 16.57 | 0.0 | 0.0 | 正常基准；在干净观测下策略可以避开危险区域。 |
| `safedreamer_cost_under` | 16.57 | 0.0 | 0.0 | 与 clean 相同，因为 clean 的真实 cost 已经为 0，且本轮是评估而非重新训练。 |
| `safedreamer_obs_noise_0.1` | 12.32 | 0.0 | 0.0 | 观测噪声降低任务收益，但本轮没有破坏安全性。 |
| `safedreamer_obs_delay_2` | 13.05 | 16.8 | 0.6 | 观测延迟导致避障决策基于过期状态。 |
| `safedreamer_action_noise_0.1` | 15.82 | 10.6 | 0.4 | 动作执行噪声使真实轨迹偏离策略计划。 |
| `safedreamer_action_delay_2` | 11.07 | 12.8 | 0.8 | 控制延迟削弱闭环修正能力，导致较高违规率。 |
| `safedreamer_hazard_blind` | 2.79 | 42.8 | 1.0 | 危险物信息被移除，安全感知通道失效。 |
| `safedreamer_lidar_blind` | 11.17 | 80.2 | 1.0 | 观测向量尾部维度被污染，产生本轮最高安全代价。 |

## 主要发现

整体结果与实验假设一致。SafeDreamer 在 clean 条件下真实 cost 为 0，说明该 checkpoint 在观测完整时可以较安全地完成任务。在安全相关攻击下，尤其是 `hazard_blind` 和 `lidar_blind`，`true_cost` 和 `violation_rate` 明显上升，说明该策略对安全信息缺失或观测污染较敏感。

`hazard_blind` 是最直接的安全攻击。它将 hazard 相关 observation slice 置零，使策略无法显式感知危险物体。结果中 5 个 episode 全部发生安全违规，平均真实 cost 上升到 42.8。

`lidar_blind` 的影响更严重，平均真实 cost 为 80.2，violation rate 为 1.0。在当前向量观测实现中，该攻击遮蔽的是 observation 的最后若干维，而不是一个物理上独立的 lidar 传感器。因此它可能同时污染 hazard 相关特征和部分机器人状态特征，这可以解释为什么它造成的安全退化强于 `hazard_blind`。

`obs_noise_0.1` 将平均 return 从 16.57 降低到 12.32，但真实 cost 仍为 0。这说明中等强度的高斯观测噪声会影响导航质量，但在本轮实验中还没有把策略推过安全边界。

`obs_delay_2`、`action_noise_0.1` 和 `action_delay_2` 产生了中等程度的安全退化。这些攻击没有完全移除安全信息，但破坏了控制器依赖的时序假设或执行假设。其中 action delay 尤其危险，因为环境实际执行的是基于旧状态规划出的动作。

`cost_under` 在本轮实验中没有可见影响。这是合理的，因为 clean 条件下真实 cost 为 0，并且当前 `osrp_vector` 评估使用已有 checkpoint，而不是在线训练新的 cost-aware 策略。cost 低报更适合用于重新训练、微调或在线适应实验，因为这些场景中 agent 会把暴露出来的 cost 当作学习信号或约束信号。

## 算法原理解释

SafeDreamer 是一种基于世界模型的强化学习方法。它学习或加载一个 latent dynamics model，将历史 observation 和 action 映射为 latent state，然后在 imagined rollout 中预测未来 reward、cost 和 state transition。策略或 planner 根据这些模型预测选择动作。

这种架构带来了几个关键攻击面：

- observation 攻击会污染世界模型推断出的 latent state；
- hazard 或 lidar blind 会在规划前移除安全关键特征；
- observation delay 会让 latent state 滞后于真实机器人状态；
- action noise 和 action delay 会导致策略选择的动作与环境实际执行的动作不一致；
- cost under-reporting 会在训练或在线适应时误导学习过程和安全约束更新。

本实验中最强的失败来自会改变策略危险物估计的攻击。这与世界模型控制的机制一致：如果模型输入中不包含危险物信息，那么 imagined rollout 就无法正确估计碰撞风险，策略自然可能选择穿过危险区域的动作。

## 合理性与局限性

作为课程项目中的鲁棒性实验，本轮结果是合理的。clean baseline 安全，而与 hazard 感知相关的攻击显著提高了 `true_cost` 和 `violation_rate`。这支持了本项目的核心论点：SafeDreamer 的安全表现强依赖可靠的安全观测。

不过，当前结果仍应视为阶段性结果，而不是最终统计结论。每个条件 5 个 episode 足以观察趋势，但不足以支撑强统计结论。最终报告建议每个条件运行 10-30 个 episode，并报告均值和标准差。

报告中还需要明确两个实现细节：

- 当前 `lidar_blind` 污染的是向量观测尾部维度，因此它不是完全真实的物理 lidar 失效模型；
- `cost_under` 在 `eval_only` 模式且 clean cost 为 0 时较弱；当训练或适应 cost-aware SafeDreamer 策略时，它会更有意义。

## 建议下一步

1. 每个条件运行 10 个 episode：

```bash
python scripts/run_safedreamer_attack_suite.py --episodes 10 --continue-on-error
```

2. 在 summary 表中加入 return 和 true cost 的 `mean +/- std`。

3. 如果时间允许，测试更强的 observation noise，例如 `0.2` 和 `0.3`。

4. 若要让 `cost_under` 成为更有效的实验，需要重新训练或微调一个会使用 exposed cost 更新策略的 cost-aware SafeDreamer 版本。

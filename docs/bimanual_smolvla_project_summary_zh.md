# 双臂 Handover + SmolVLA 实验总结

**平台：** LeRobot + 双 SO101（`bi_so101_follower`）  
**任务：** 抓取双面胶卷 → 递给另一臂 → 放入黄色长方形塑料盘  
**周期：** 2026 年 7 月  
**工作侧重：** 数据采集、模型训练、真机部署调试、评估方法

---

## 1. 项目概述（面试一句话）

在双 SO101 机器人上搭建完整的 **视觉-语言-动作（VLA）** 双臂操作链路：采集 80 条遥操作 demo（6.5 万帧、12 维关节、3 路相机），从单臂 **SmolVLA** 基座微调，并部署到真机。通过 **offline MAE 指标** 和 **真机闭环调试**，定位部署瓶颈为 **过拟合、控制环参数、任务描述不完整**，而非传感器校准或场景变化。

**面试 elevator pitch：**  
「微调 SmolVLA 完成双臂 handover；用 MAE 离线评估和控制参数分析诊断真机失败，区分模型能力上限与工程问题。」

---

## 2. 系统架构

```
遥操作 demo（80 ep，3 相机，12 维关节）
        ↓
LeRobot 数据集（bimanual_handover）
        ↓
SmolVLA 微调（smolvla_base → 12 维 action，3 相机）
        ↓
Checkpoint + preprocessor/postprocessor（归一化/反归一化）
        ↓
真机：bi_so101_follower + lerobot-record 推理循环
```

| 组件 | 配置 |
|------|------|
| 机器人 | 左右 SO101 follower，各 6 关节 + 夹爪 |
| 相机 | 俯视 UGREEN、左腕 YHTek、右侧 Sonix → 重命名为 camera1/2/3 |
| action / state | 12 维（左 6 + 右 6），模型内部 pad 到 32 |
| 训练 | 2× GPU，batch 16，150k step（≈37 epoch），loss → 0.001 |
| 推理（调参后） | `n_action_steps=5`，`max_relative_target=50` |

---

## 3. 关键技术发现

### 3.1 Checkpoint 元数据 vs 实际训练维度

从单臂 `smolvla_base`（state=6）微调时：

- **`output_features.action`** 正确从数据集更新为 **12 维**。
- **`input_features.observation.state`** 在 `config.json` 里仍显示 **6 维**，因为 `factory.py` 仅在 `input_features` 为空时才用数据集覆盖。
- **训练实际仍喂 12 维 state**；normalizer 的 mean/std 通过训练 override 使用 12 维数据集统计量。
- **模型 `state_proj`** 固定接收 pad 后的 **32 维**（`max_state_dim=32`）。

**面试要点：** 区分 **config 元数据 bug** 与 **运行时 tensor 形状**；不能只看 `config.json`，要核对 safetensors 和 batch 维度。

### 3.2 Action 存的是双臂，不是单臂

Checkpoint 中 action 为 **12 维**（左右臂关节目标）；postprocessor unnormalizer 也是 `[12]`。这条链路是正确的。

### 3.3 任务文字 vs 真实任务

| 层面 | 内容 |
|------|------|
| 采集 / 训练 | `"Grab and handover the object to the other arm"` |
| 真实操作 | 抓双面胶卷 → 交接 → 放入 **黄色塑料盘** |

SmolVLA 会 tokenize 任务文字，但 **模仿学习主要靠视觉**。泛化 task 没写「放入黄盘」，第三步只能靠画面学。**推理时不要改成更长描述**（训练没见过会 OOD），除非重新采集并重训。

### 3.4 Offline MAE 评估方法

在训练集上抽 5 帧，逐帧计算：

```python
mae = mean(|pred_action - gt_action|)      # 12 关节，归一化电机单位
delta = mean(|pred_action - current_state|) # 运动意图（是否想动）
```

| 帧 | MAE | 解读 |
|----|-----|------|
| 0 | 0.72 | 起手姿态拟合好 |
| 1000–5000 | ~1.0 | 可接受 |
| 10000–20000 | 3.4–3.6 | handover 中段偏弱 |

**单位：** 不是角度（°），是 SO101 归一化值（身体关节约 −100~+100，夹爪 0~100）。

**局限：** 单帧、`policy.reset()`、仅在训练分布上测。适合 **健康检查**，不能代表任务成功率。

### 3.5 真机表现

现象：右臂缓慢抬到中间高度后 **悬停、基本不动**。场景、光照、相机均未改变。

**已排除：** normalizer 6/12 维问题（offline MAE 和右臂误差正常）、场景不一致。

**最可能原因（按优先级）：**

1. **过拟合**：150k step、loss≈0.001 → 策略输出接近当前 state（「保持」）。
2. **闭环发散**：绝对关节角策略，小误差在中途累积。
3. **起手关节角** 与 demo 第 0 帧有细微偏差。
4. **控制参数**：`n_action_steps`（重规划频率）、`max_relative_target`（单步限速）。

### 3.6 控制参数（部署调参）

| 参数 | 作用层 | 效果 |
|------|--------|------|
| `N_ACTION_STEPS` | 模型重规划 | 越小越跟手（5 ≈ 每 0.17s @ 30fps 看一次新图） |
| `MAX_RELATIVE_TARGET` | 机器人安全裁剪 | 越大单步越快（50 vs 25） |

两者 **正交**：一个管「多久重新想」，一个管「每步最多动多少」。

---

## 4. 产出与代码改动

| 文件 | 用途 |
|------|------|
| `get-data-bimanual.sh` | 双臂遥操作采集，3 相机 |
| `train_smolvla_bimanual.sh` | 微调：rename_map、`tolerance_s=0.05`、`empty_cameras=0` |
| `infer_smolvla_bimanual.sh` | 真机 / offline / viz；默认调参 |
| `src/lerobot/robots/utils.py` | `max_relative_target` int→float 修复 |
| `bi_so101_follower` config | max_relative_target 类型改为 float |

---

## 5. 结果汇总

| 指标 | 数值 |
|------|------|
| 数据集 | 80 episodes，65,301 frames，12 维 action/state |
| 训练 | 150k steps，最终 loss ~0.001 |
| Offline MAE（150k） | 抽样帧 0.7–3.6，均值约 2 |
| 真机 | 部分成功：抬臂至中段后停滞 |
| 部署改进 | 从「几乎不动」→「能抬到中间」（调参后） |

---

## 6. 经验教训（面试可讲）

### 调试方法论

1. **分开看 offline 与真机失败模式** — offline MAE 低不等于任务成功；用 `delta`（pred vs state）判断是否「装死」。
2. **Checkpoint 扫描** — 越晚不一定越好；训练 loss 极低（0.001）常损害闭环控制。
3. **读全栈** — config JSON、preprocessor stats、`factory.py` 特征 wiring、机器人 action 裁剪、VLA action 队列。
4. **单变量排查** — 确认场景未变后，将分析转向过拟合与控制参数。

### 机器人 / ML

1. **绝对关节角策略** 即使场景相同，闭环误差累积仍会导致中段失败。
2. **VLA 任务描述** 应覆盖完整子任务；否则后半段全靠视觉，学习难度大。
3. **双臂 handover** 对小体量 VLA 是难任务；同一数据集值得并行试 **GR00T**。

### 工程细节

1. **`tolerance_s=0.05`** 才能加载 3 路视频时间戳对齐的全部 80 ep。
2. **双臂校准文件** 与单臂分开；电机 EEPROM 写入最后一次校准。
3. 训练时 **normalizer stats override** 与推理加载路径不完全一致，建议显式对齐。

---

## 7. 后续建议

| 优先级 | 行动 |
|--------|------|
| P0 | 真机试 **50k / 100k** checkpoint（别用 150k） |
| P1 | Offline 对比 50k / 100k / 150k MAE |
| P2 | 再采 30+ 条，使用 **完整 task 描述** |
| P3 | 重训 250k step；部署中期 checkpoint |
| P4 | 同一数据集并行训练 **GR00T** |
| P5 | 修 `factory.py`，微调时同步 `input_features` |

---

## 8. 面试 Q&A 示例

**问：最难的调试时刻是什么？**  
答：模型加载成功但机械臂几乎不动。我从 checkpoint 元数据（state 6 vs 12）、normalizer、offline MAE 一直查到控制参数。offline MAE 约 1–3 说明模型没坏，是过拟合导致闭环下输出接近静止。

**问：没有仿真怎么评估？**  
答：在训练集上做 offline 推理（12 关节 MAE + 夹爪分项），真机用 `lerobot-record` 录 eval；配合 checkpoint 扫描和运动意图指标 |pred − state|。

**问：如果重做会改什么？**  
答：（1）采集第一天就用完整 task 描述；（2）训练过程中多存、多评 checkpoint；（3）双臂任务更早启动 GR00T；（4）修 training factory 的特征元数据。

**问：对 VLA 部署的理解？**  
答：训练 loss 和 offline 精度不够。部署还要调重规划频率、单步限速、起手姿态对齐、checkpoint 早停——绝对关节角真机尤其敏感。

---

## 9. 简历 / 项目经历（STAR）

| | |
|--|--|
| **S（背景）** | 需让双臂机器人完成「抓胶卷 → 交接 → 放黄盘」，从单臂 VLA 扩展到 12 维双机系统。 |
| **T（任务）** | 搭建采集→训练→真机推理全链路，使策略在实机上完成 handover。 |
| **A（行动）** | 采 80 条三相机 demo；编写 bimanual 训练/推理脚本；修复校准与类型 bug；设计 offline MAE 评估；调 `n_action_steps` 与 `max_relative_target`；分析过拟合与闭环控制。 |
| **R（结果）** | Offline MAE ~1–3；真机从「几乎不动」进步到「右臂抬至中段」；明确瓶颈与补数据、重训、GR00T 路线。 |

**简历 bullet 示例（可直接粘贴）：**

- 基于 LeRobot + 双 SO101 搭建 SmolVLA 双臂 handover 流水线（80 demo / 6.5 万帧 / 3 相机），完成微调与真机部署。
- 设计 offline MAE + 夹爪分项误差评估，区分模型过拟合与 normalizer / 场景问题。
- 定位真机「中段悬停」为过拟合与闭环控制参数问题，调优 `n_action_steps` 与 `max_relative_target`，改善运动响应。
- 发现并分析 `factory.py` 在单臂→双臂微调时 state 元数据未同步的根因。

---

## 10. 常用命令

```bash
# Offline 评估
bash infer_smolvla_bimanual.sh offline

# 更早 checkpoint 真机推理
CHECKPOINT=outputs/train/smolvla_bimanual_handover/checkpoints/050000/pretrained_model \
  CUDA_VISIBLE_DEVICES=1 bash infer_smolvla_bimanual.sh robot

# 可视化训练 demo
bash infer_smolvla_bimanual.sh viz

# 更长训练
STEPS=250000 bash train_smolvla_bimanual.sh
```

---

*文档来源：SmolVLA 双臂 handover 实验，2026 年 7 月。*

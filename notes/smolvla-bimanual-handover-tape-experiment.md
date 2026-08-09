# SmolVLA · Bimanual Pass（双面胶 → 交接 → 黄盘）实验总结

面向 **技术学习笔记 / 求职面试**。模板对齐 `notes/smolvla-bimanual-stack-bowls-experiment.md`（含 Episodes / Steps / Epochs、Duration、Latency、Hz、SR、失败模式）。

**主线（当前评测）：** 4 相机数据集 `bimanual_handover_4cam_20260717_150424`  
**前身：** 3 相机 `bimanual_handover`（过拟合 / 悬停教训）

相关脚本：
- 采集：`get-data-bimanual.sh`
- 训练：`train_smolvla_bimanual_4cam_20260717.sh`（旧 3cam：`train_smolvla_bimanual.sh`）
- 推理：`infer_smolvla_bimanual_4cam_20260717_ep20.sh`（另有 ep18/22/25/39）

---

## 1. 任务与设置

| 项目 | 内容 |
|------|------|
| 任务指令（4cam） | `Grasp the double-sided tape roll, hand it over to the other arm, and place it in the yellow rectangular plastic tray.` |
| 成功标准 | 第一臂抓住双面胶 → 递给另一臂 → 放入黄盘；时限内；**零人工干预** |
| 机器人 | 双臂 SO101 follower（**12 DoF**）+ 双 leader 遥操作 |
| 相机（4cam） | `top_left` / `top_right` / `left` / `right`，MJPG @ 30fps |
| 策略 | SmolVLA（~450M，`train_expert_only` 路线） |
| 相机映射 | `top_left→camera1` … `right→camera4` |
| 环境 | `conda: lerobot`，训练多为 **2×3090**，`BATCH_SIZE=4` → effective **8** |

**子技能链：** `grasp(tape)` → `handover` → `place(yellow tray)`。  
经验上瓶颈在 **第一臂抓取**，不在交接接爪。

---

## 2. 数据（Episodes / Frames）

### 2.1 主库：4 相机（20260717_150424）

| 指标 | 数值 |
|------|------|
| Dataset | `/home/rxn/datasets/bimanual_handover_4cam_20260717_150424` |
| Episodes | **89** |
| Frames | **47,074** |
| FPS | 30 |
| 平均时长 | **~17.6 s/ep** |
| Action / State | 各 **12** 维 |
| 质量体检 | 静止约 15%；自动体检未大量标红 |

### 2.2 前身：3 相机

| 指标 | 数值 |
|------|------|
| Dataset | `/home/rxn/datasets/bimanual_handover` |
| Episodes | **80** |
| Frames | **65,301** |
| 相机 | `top` / `left` / `right`（3 路） |
| 文本（偏泛） | `Grab and handover the object to the other arm`（真实含入黄盘，文字未写全） |

---

## 3. 训练（Steps / Epoch / Batch）

### 3.1 4cam（主结果）

| 超参 | 数值 |
|------|------|
| 计划 Steps | **200,000** |
| 实际 | 手动停在 **~120,000**（KeyboardInterrupt） |
| Batch / GPU | 4；2 GPU → effective **8** |
| 按帧估算 data-epoch @120k | \(120000\times8/47074\approx\) **20.4** |
| 日志 `epch` @120k | **≈ 40.7**（tracker 约为上式 ×2；与叠碗相同口径问题） |
| 脚本/面试所称「~20 ep」ckpt | **`060000`**（\(60000\times16/47074\approx20.4\) 日志口径；\(60000\times8/47074\approx10.2\) data-epoch） |
| 甜区（经验） | **`060000`–`080000`**（称 ~20–27 log-epoch）；再往后偏过拟合 |
| 停训时 loss | ~**0.027–0.029** |
| `updt_s` | ~**0.16–0.17 s/step** |

**Checkpoint：**
```text
outputs/train/smolvla_bimanual_4cam_20260717_150424/checkpoints/
  060000/pretrained_model   # 主推真机（ep20 脚本默认）
  070000 / 080000           # 邻近甜区
  120000 / last             # 停训点，偏晚
```

### 3.2 3cam（对照教训）

| 项 | 值 |
|----|-----|
| Steps | **150,000**（训满） |
| loss | → **~0.001**（过拟合强烈） |
| 真机 | 右臂抬起后 **悬停**；应用中期 ckpt（50k/100k）而非 150k |
| Offline | 交接中段 MAE 明显变差（健康检查，非 SR） |

**统一结论：** 双臂长 horizon BC，**中期 ckpt + 真机 SR 选模**，不要只看最终 loss。

---

## 4. 推理设置（Duration / Latency / Hz）

| 项 | 默认值（`infer_…_ep20.sh`） |
|----|---------------------------|
| Ckpt | `060000`（~20 log-epoch 甜区） |
| `EPISODE_TIME_S` / Duration | **65 s**（采集同默认） |
| `RESET_TIME_S` | 5 s |
| 控制 FPS | **30**（周期预算 ≈ **33.3 ms**） |
| `n_action_steps` | **5**（训练 chunk 50；真机缩短更跟手） |
| `max_relative_target` | **50** |
| 有效重规划 | 约每 \(5/30\approx\) **0.17 s** 用新观测再规划一段 chunk |

**Latency / Hz（同架构参考：蓝块任务 3090 上测；本任务未单独重测）：**

| 指标 | 数值 |
|------|------|
| Policy `select_action` | ~**148 ms** |
| 等效策略刷新 | ~**6.8 Hz** |
| 控制环目标 | **30 Hz** |

→ 单步推理 **慢于** 33 ms 预算，必须靠 **action chunk**；不能把 148 ms 和 episode Duration 65 s 混为一谈。

面试分清三层：

| 名词 | 含义 | 本实验量级 |
|------|------|------------|
| Policy latency | 一次前向 | ~150 ms（~7 Hz） |
| Control rate | 发关节指令 | 30 Hz |
| Episode duration | 评测时限 | **65 s** |

---

## 5. 真机评测（Success Rate）

**协议：** \(SR=\#成功/N\)。成功 = 抓胶带 → 交接 → 入黄盘；时限内；零干预。

### 5.1 小样本（~20 ep 甜区 ckpt，定性）

| 项 | 值 |
|----|-----|
| Trials | ≈ **3** |
| 成功 | ≈ **2** |
| **SR** | **≈ 2/3 ≈ 67%**；可报 **≥ 50%**（\(N\) 小，正式建议 \(N\ge10\)） |
| 条件规律 | **只要第一臂抓住胶带，后续 pass + 入盘基本成功** |

### 5.2 失败模式（本实验核心观察）

#### A. 瓶颈在第一臂抓取，不在交接

> 容易失败的地方 **不是** 另一臂去接双面胶，而是 **第一臂去夹双面胶** 时出现 **空抓（空夹）**。  
> 一旦夹住并进入 handover，出错明显变少。

**解读：**
- 接触式抓取对位容差小（卷材圆柱、视角遮挡、深度估计弱）。
- Handover / place 更偏「已持物后的运动原语」，示教更一致，BC 更好学。
- 面试可说：长任务 SR 常由 **最难子技能** 决定；这里是 **initial grasp**，不是 bimanual pass 本身。

#### B. 「第一臂抓住 ≈ 整条成功」

条件成功视角：

\[
P(\text{full success}) \approx P(\text{first grasp}) \times P(\text{pass+place}\mid\text{grasp})
\]

经验上 \(P(\text{pass+place}\mid\text{grasp})\) 很高 → 提升 SR 应优先 **加抓取段数据 / 清洗空抓 demo / 腕相机对齐**，而不是只堆 handover 数据。

#### C. 与 3cam 悬停对比

| 版本 | 典型失败 |
|------|----------|
| 3cam 训满 150k | 中途 **悬停**（过拟合「保持」） |
| 4cam 甜区 ~20 ep | **空抓**；抓住后链路较稳 |

说明：修好过拟合与控制参数后，能力上限暴露在 **感知-抓取**，而不是整段不动。

---

## 6. Offline vs Robot

| 模式 | 作用 | 局限 |
|------|------|------|
| Offline MAE | 查权重/归一化是否健康 | 单帧 reset，**不是**任务 SR |
| Robot SR | 唯一可信任务指标 | 需固定 ckpt、时限、零干预、足够 \(N\) |

3cam 上交接中段 MAE 变差，与真机脆弱段一致，但仍不能代替 SR。

---

## 7. 工程与部署要点

| 点 | 说明 |
|----|------|
| Task 文本 | 4cam 已写全「胶带→交接→黄盘」；3cam 文本偏泛，第三步靠视觉学 |
| `tolerance_s=0.05` | 双臂视频时间戳漂移必开 |
| `n_action_steps` vs `max_relative_target` | 正交：重规划频率 vs 单步限幅 |
| 硬件 | 曾遇右臂 Feetech **id=5 Input voltage error**，打断评测 |
| 相机占用 / USB 掉线 | 与叠碗相同：推理前确认 `by-id` |

---

## 8. 面试怎么讲（60–90 秒）

> 我做了双臂 SO101 双面胶 handover：抓胶带、递给另一臂、放入黄盘。4 相机 89 ep / 4.7 万帧，SmolVLA 训到约 12 万 step 手动停，真机用约 20 epoch 甜区 ckpt（`060000`）。控制 30 Hz、chunk `n_action_steps=5`，策略前向约 150 ms 所以必须 chunk。小样本大约 3 次里成功 2 次、SR 能到 50% 以上。失败几乎都在第一臂空抓；只要抓住，交接和入盘很少错。说明双臂 BC 的瓶颈是接触抓取，不是 pass 本身；数据与评测应围着 first-grasp 加强，并用中期 checkpoint 避免过拟合悬停。

**追问答法：**

1. **为何不是交接难？**  
   示教里持物后运动更一致；空抓是开环对位误差，视觉反馈弱。

2. **如何提高 SR？**  
   多采成功抓取、删空抓段、保证腕相机可见胶带、正式 \(N\ge10\) 扫 18/20/25 ep ckpt。

3. **Duration vs Latency？**  
   Duration=65 s 任务时限；Latency≈150 ms 单次推理；控制 30 Hz + chunk。

4. **和叠碗对比？**  
   叠碗瓶颈在末段放置 + 阶段门控；本任务瓶颈在首段 grasp。都说明要做 **阶段级 failure analysis**，不能只报一个 SR。

---

## 9. 数字速查

```text
Task           : tape grasp → handover → yellow tray (bimanual)
Data (4cam)    : 89 ep / 47,074 fr @ 30 FPS / 12-DoF / 4 cams
Train          : plan 200k, stop ~120k; sweet ckpt 060000–080000
Infer Duration : 65 s/ep  |  FPS 30  |  n_action_steps=5
Latency ref    : ~148 ms/select_action (~6.8 Hz) on 3090
Control budget : 33 ms/frame @ 30 Hz → need action chunks
Robot SR       : ~2/3 (≥50%), N≈3 informal; first-grasp gated
Fail mode      : empty grasp on first arm; pass rarely fails if grasped
3cam lesson    : 150k loss~0.001 → hover; use mid ckpt
```

---

## 10. 路径速查

| 用途 | 路径 |
|------|------|
| 4cam 数据 | `/home/rxn/datasets/bimanual_handover_4cam_20260717_150424` |
| 3cam 数据 | `/home/rxn/datasets/bimanual_handover` |
| 4cam 训练 | `outputs/train/smolvla_bimanual_4cam_20260717_150424` |
| 甜区权重 | `.../checkpoints/060000/pretrained_model` |
| 推理 | `bash infer_smolvla_bimanual_4cam_20260717_ep20.sh robot` |
| 日志 | `logs/smolvla_bimanual_4cam_20260717_150424.log` |
| 旧总结 | `docs/bimanual_smolvla_project_summary_zh.md` |
| 叠碗对照 | `notes/smolvla-bimanual-stack-bowls-experiment.md` |

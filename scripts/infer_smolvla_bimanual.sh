#!/bin/bash
# =============================================================================
# SmolVLA 推理（bimanual_handover 双臂 handover）
# Usage:
#   conda activate lerobot
#   bash infer_smolvla_bimanual.sh robot      # 真机推理
#   bash infer_smolvla_bimanual.sh offline    # 数据集离线推理
#   bash infer_smolvla_bimanual.sh preview    # Rerun 三相机预览（无需机械臂）
#   bash infer_smolvla_bimanual.sh cameras    # 列出相机/串口
#   bash infer_smolvla_bimanual.sh viz        # 可视化训练集或最近一次 eval
#
# 指定 checkpoint:
#   CHECKPOINT=.../checkpoints/100000/pretrained_model bash infer_smolvla_bimanual.sh robot
#   CHECKPOINT=.../checkpoints/50000/pretrained_model EPISODE_TIME_S=65 bash infer_smolvla_bimanual.sh robot
#
# 其他覆盖:
#   MAX_RELATIVE_TARGET=25 CUDA_VISIBLE_DEVICES=1 NUM_EPISODES=3 bash infer_smolvla_bimanual.sh robot  # 更保守
# =============================================================================
set -euo pipefail

MODE="${1:-robot}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/bimanual_handover_4cam}"
DATA_REPO="${DATA_REPO:-my_bimanual/handover_4cam}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_bimanual_handover_4cam}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# ---------- 双臂 follower ----------
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"

# ---------- 四相机 ----------
TOP_LEFT_CAM="${TOP_LEFT_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
TOP_RIGHT_CAM="${TOP_RIGHT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
LEFT_CAM="${LEFT_CAM:-/dev/v4l/by-id/usb-am_camera_wrist_left_am_camera_wrist_left-video-index0}"
RIGHT_CAM="${RIGHT_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
TOP_CAM="${TOP_CAM:-${TOP_RIGHT_CAM}}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
TOP_LEFT_CAM_WIDTH="${TOP_LEFT_CAM_WIDTH:-800}"
TOP_LEFT_CAM_HEIGHT="${TOP_LEFT_CAM_HEIGHT:-480}"
CAM_WARMUP_S="${CAM_WARMUP_S:-8}"
FPS="${FPS:-30}"

ROBOT_CAMERAS="{ top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, left: {type: opencv, index_or_path: \"${LEFT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, right: {type: opencv, index_or_path: \"${RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}} }"

# top_left/top_right/left/right -> camera1/2/3/4（与 train_smolvla_bimanual.sh 一致）
RENAME_MAP='{"observation.images.top_left":"observation.images.camera1","observation.images.top_right":"observation.images.camera2","observation.images.left":"observation.images.camera3","observation.images.right":"observation.images.camera4"}'
TOLERANCE_S="${TOLERANCE_S:-0.05}"

NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-65}"
RESET_TIME_S="${RESET_TIME_S:-5}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"   # 默认 5（更跟手）；训练 chunk 50，真机可试 5~10
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_bimanual_handover_4cam}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy" 2>/dev/null || {
    echo "ERROR: SmolVLA not installed. pip install -e \".[smolvla,feetech]\""
    exit 1
}

if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    ls -la "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
    exit 1
fi

CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
    CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

DATASET_TASK=$(python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}", tolerance_s=${TOLERANCE_S})
print(ds.meta.tasks.index[0])
PY
)

echo "=============================="
echo "SmolVLA Bimanual Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "Dataset:    ${DATA_ROOT}"
echo "Eval out:   ${EVAL_ROOT}"
echo "Task:       ${DATASET_TASK}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Duration:   ${NUM_EPISODES} ep × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset)"
echo "n_action_steps: ${N_ACTION_STEPS} (训练默认 50)  max_relative_target: ${MAX_RELATIVE_TARGET}"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
    echo "Serial (/dev/serial/by-id):"
    ls -la /dev/serial/by-id/ 2>/dev/null || true
    echo ""
    echo "Cameras (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || true
    echo ""
    echo "Defaults:"
    echo "  LEFT_FOLLOWER_PORT=${LEFT_FOLLOWER_PORT}"
    echo "  RIGHT_FOLLOWER_PORT=${RIGHT_FOLLOWER_PORT}"
    echo "  TOP_LEFT_CAM=${TOP_LEFT_CAM}"
    echo "  TOP_RIGHT_CAM=${TOP_RIGHT_CAM}"
    echo "  LEFT_CAM=${LEFT_CAM}"
    echo "  RIGHT_CAM=${RIGHT_CAM}"
    echo ""
    lerobot-find-cameras opencv || true
    exit 0
fi

if [ "${MODE}" = "preview" ]; then
    echo "Checking cameras..."
    python - <<PY
import sys, cv2, time

def check(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                shape = frame.shape
                break
            time.sleep(0.05)
        ok = shape is not None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({src})")
    return ok

cams = [
    ("top_left", "${TOP_LEFT_CAM}"),
    ("top_right", "${TOP_RIGHT_CAM}"),
    ("left", "${LEFT_CAM}"),
    ("right", "${RIGHT_CAM}"),
]
if not all(check(n, s) for n, s in cams):
    sys.exit(1)
PY

    echo ""
    echo "Rerun GUI：四合一 2x2 top_left|top_right / left|right，Ctrl+C 退出"
    echo ">>> 3 秒后开始..."
    sleep 3

    python - <<PY
import time
import cv2
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [
    ("top_left", "${TOP_LEFT_CAM}"),
    ("top_right", "${TOP_RIGHT_CAM}"),
    ("left", "${LEFT_CAM}"),
    ("right", "${RIGHT_CAM}"),
]
W, H = ${CAM_WIDTH}, ${CAM_HEIGHT}

def open_cam(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL: {name} ({src})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(45):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK: {name} {frame.shape} ({src})")
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(f"FAIL: {name} no frame ({src})")

init_rerun(session_name="smolvla_bimanual_cam_preview")
caps = {name: open_cam(name, src) for name, src in CAMS}

def label_rgb(frame, text):
    out = frame.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out

print("Rerun 已开。主画面: observation.all (2x2 四合一)")
try:
    while True:
        t0 = time.perf_counter()
        obs = {}
        labeled = {}
        for name, cap in caps.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != W or rgb.shape[0] != H:
                    rgb = cv2.resize(rgb, (W, H))
                obs[name] = rgb
                labeled[name] = label_rgb(rgb, name)
        if len(labeled) == 4:
            row_top = cv2.hconcat([labeled["top_left"], labeled["top_right"]])
            row_bot = cv2.hconcat([labeled["left"], labeled["right"]])
            obs["all"] = cv2.vconcat([row_top, row_bot])
        log_rerun_data(observation=obs, action={})
        time.sleep(max(0, 1 / ${FPS} - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    for cap in caps.values():
        cap.release()
    rr.rerun_shutdown()
PY
    exit 0
fi

if [ "${MODE}" = "viz" ]; then
    if [ -d "${EVAL_ROOT}" ] && [ -n "$(ls -A "${EVAL_ROOT}" 2>/dev/null)" ]; then
        LATEST_EVAL=$(ls -td "${EVAL_ROOT}"/*/ 2>/dev/null | head -1)
        LATEST_EVAL="${LATEST_EVAL%/}"
        echo "Opening eval recording: ${LATEST_EVAL}"
        lerobot-dataset-viz \
          --repo-id="upna/eval_smolvla_bimanual_handover" \
          --root="${LATEST_EVAL}" \
          --episode-index=0 \
          --tolerance-s="${TOLERANCE_S}" \
          --mode=local
    else
        echo "Opening training dataset (episode 0)..."
        lerobot-dataset-viz \
          --repo-id="${DATA_REPO}" \
          --root="${DATA_ROOT}" \
          --episode-index=0 \
          --tolerance-s="${TOLERANCE_S}" \
          --mode=local
    fi
    exit 0
fi

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import json
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

checkpoint = "${CHECKPOINT}"
data_root = "${DATA_ROOT}"
repo_id = "${DATA_REPO}"
rename_map = json.loads('''${RENAME_MAP}''')
tolerance_s = ${TOLERANCE_S}

cfg = PreTrainedConfig.from_pretrained(checkpoint)
cfg.pretrained_path = checkpoint
cfg.device = "cuda"

ds = LeRobotDataset(repo_id=repo_id, root=data_root, tolerance_s=tolerance_s)
policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=rename_map)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=checkpoint,
    dataset_stats=rename_stats(ds.meta.stats, rename_map),
    preprocessor_overrides={
        "rename_observations_processor": {"rename_map": rename_map},
    },
)
policy.eval()

print("Running offline inference on bimanual_handover dataset...")
for idx in [0, 1000, 5000, 10000, 20000]:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    batch = preprocessor({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()})
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    gt = sample["action"].float()
    pred = action.cpu().float().view(-1)
    mae = (pred - gt).abs().mean().item()
    lg_mae = (pred[5] - gt[5]).abs().item()
    rg_mae = (pred[11] - gt[11]).abs().item()
    print(f"  frame {idx:6d}  mae={mae:.4f}  left_grip={lg_mae:.3f}  right_grip={rg_mae:.3f}")
print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    if [ ! -e "${LEFT_FOLLOWER_PORT}" ] || [ ! -e "${RIGHT_FOLLOWER_PORT}" ]; then
        echo "ERROR: Follower port not found."
        echo "  left:  ${LEFT_FOLLOWER_PORT} $([ -e "${LEFT_FOLLOWER_PORT}" ] && echo OK || echo MISSING)"
        echo "  right: ${RIGHT_FOLLOWER_PORT} $([ -e "${RIGHT_FOLLOWER_PORT}" ] && echo OK || echo MISSING)"
        ls /dev/serial/by-id/ 2>/dev/null || true
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (need ~${MIN_FREE_MIB} MiB)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: Not enough GPU memory on GPU ${PHYS_GPU}."
            nvidia-smi
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
        exit 1
    }

    echo "Left follower:  ${LEFT_FOLLOWER_PORT}"
    echo "Right follower: ${RIGHT_FOLLOWER_PORT}"
    echo "Top left cam:   ${TOP_LEFT_CAM}"
    echo "Top right cam:  ${TOP_RIGHT_CAM}"
    echo "Left cam:       ${LEFT_CAM}"
    echo "Right cam:      ${RIGHT_CAM}"
    echo ""
    echo "Checking cameras..."
    python - <<PY
import sys, cv2, time

def check(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                shape = frame.shape
                break
            time.sleep(0.05)
        ok = shape is not None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({src})")
    return ok

failed = not check("top_left", "${TOP_LEFT_CAM}") or not check("top_right", "${TOP_RIGHT_CAM}") or not check("left", "${LEFT_CAM}") or not check("right", "${RIGHT_CAM}")
if failed:
    print("ERROR: camera check failed. Run: bash infer_smolvla_bimanual.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting SmolVLA bimanual inference in 3s (Ctrl+C to abort)..."
    sleep 3

    EVAL_DATA_ROOT="${EVAL_ROOT}/$(date +%Y%m%d_%H%M%S)"
    echo "Eval save: ${EVAL_DATA_ROOT}"

    lerobot-record \
      --robot.type=bi_so101_follower \
      --robot.left_arm_port="${LEFT_FOLLOWER_PORT}" \
      --robot.right_arm_port="${RIGHT_FOLLOWER_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.left_arm_max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.right_arm_max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="${ROBOT_CAMERAS}" \
      --display_data=true \
      --play_sounds="${PLAY_SOUNDS}" \
      --dataset.repo_id="upna/eval_smolvla_bimanual_handover" \
      --dataset.root="${EVAL_DATA_ROOT}" \
      --dataset.single_task="${DATASET_TASK}" \
      --dataset.fps="${FPS}" \
      --dataset.num_episodes="${NUM_EPISODES}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.push_to_hub=false \
      --dataset.rename_map="${RENAME_MAP}" \
      --policy.path="${CHECKPOINT}" \
      --policy.device=cuda \
      --policy.n_action_steps="${N_ACTION_STEPS}" \
      2>&1 | python -u -c "
import sys
phase = 'init'
shown = False
for raw in sys.stdin:
    line = raw.rstrip('\n')
    if 'Recording episode' in line:
        phase = 'record'
        shown = False
        print(); print('=' * 62)
        print('  SmolVLA bimanual 推理中 (${EPISODE_TIME_S}s)')
        print('=' * 62); print(line, flush=True)
    elif 'Reset the environment' in line:
        phase = 'reset'
        shown = False
        print(); print('-' * 62)
        print('  RESET 复位 (${RESET_TIME_S}s)')
        print('-' * 62); print(line, flush=True)
    elif 'No policy or teleoperator' in line and phase == 'reset':
        if not shown:
            print('  (reset 阶段日志已隐藏)', flush=True)
            shown = True
    else:
        print(line, flush=True)
"

else
    echo "Usage: bash infer_smolvla_bimanual.sh [robot|offline|preview|cameras|viz]"
    exit 1
fi

echo "=============================="
echo "Inference finished"
echo "=============================="

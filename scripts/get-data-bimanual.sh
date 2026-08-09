#!/bin/bash
# =============================================================================
# SO101 双臂遥操作数据采集
# Usage:
#   bash get-data-bimanual.sh              # 开始采集（4 相机，新数据集）
#   bash get-data-bimanual.sh cameras      # 列出相机
#   bash get-data-bimanual.sh preview      # 旧 3 路预览 top|left|right（不含 top_left）
#   bash get-data-bimanual.sh preview4     # Rerun 四合一 top_left|top_right / left|right
#   bash get-data-bimanual.sh rerun        # Rerun + 接 follower 调相机（Ctrl+C 退出）
#   bash get-data-bimanual.sh viz          # 可视化已采集数据
#
# 覆盖参数示例:
#   NUM_EPISODES=10 bash get-data-bimanual.sh
#   DATA_ROOT=/home/rxn/datasets/my_bimanual bash get-data-bimanual.sh
#   RESUME=1 DATA_ROOT=/home/rxn/datasets/bimanual_handover_4cam_20250716_143000 bash get-data-bimanual.sh
#   COLLECTION_DATE=20250716_143000 bash get-data-bimanual.sh  # 自定义日期后缀
#   # 旧 3 相机数据集路径:
#   DATA_ROOT=/home/rxn/datasets/bimanual_handover bash get-data-bimanual.sh viz
#
# 端口映射（by-id，重启后不变）:
#   左 follower  → ACM2 (5AE6083854)
#   右 follower  → ACM3 (5AB9065873)
#   左 leader    → ACM1 (5AE6084864)
#   右 leader    → ACM0 (5AB9065804)
#
# 校准说明:
#   双臂校准文件与单臂分开存储，不会覆盖 so101_follower.json / so101_leader.json。
#   ACM0/ACM1（原单臂）若已校好，首次双臂连接时按 Enter 复用即可；
#   ACM2/ACM3（新臂）需要完整校准一次。
#   切回单臂 bash get-data.sh 时如有 mismatch 提示，按 Enter 恢复即可。
# =============================================================================
set -euo pipefail

MODE="${1:-record}"

PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"

# ---------- 机器人 / 遥操作（双臂）----------
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065804-if00}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"
TELEOP_ID="${TELEOP_ID:-bimanual_leader}"

# ---------- 相机（by-id 路径，4 路）----------
# top_left / top_right / left / right（与 preview4 布局一致）
TOP_LEFT_CAM="${TOP_LEFT_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
TOP_RIGHT_CAM="${TOP_RIGHT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
LEFT_CAM="${LEFT_CAM:-/dev/v4l/by-id/usb-am_camera_wrist_left_am_camera_wrist_left-video-index0}"
RIGHT_CAM="${RIGHT_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
# 兼容旧变量名（top = top_right）
TOP_CAM="${TOP_CAM:-${TOP_RIGHT_CAM}}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
# Generic Web (top_left) 硬件只认 800x480，设 640 会报错
TOP_LEFT_CAM_WIDTH="${TOP_LEFT_CAM_WIDTH:-800}"
TOP_LEFT_CAM_HEIGHT="${TOP_LEFT_CAM_HEIGHT:-480}"
# 四相机统一 30fps（UGREEN 不支持 15）
FPS="${FPS:-30}"
CAM_WARMUP_S="${CAM_WARMUP_S:-8}"

ROBOT_CAMERAS="{ top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, left: {type: opencv, index_or_path: \"${LEFT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, right: {type: opencv, index_or_path: \"${RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}} }"

# ---------- 数据集（4 相机新数据集，与旧 3 相机 bimanual_handover 分开）----------
# record 模式默认路径带时间戳；viz/续录需显式指定 DATA_ROOT
SINGLE_TASK="${SINGLE_TASK:-Grasp the double-sided tape roll, hand it over to the other arm, and place it in the yellow rectangular plastic tray.}"
NUM_EPISODES="${NUM_EPISODES:-60}"
EPISODE_TIME_S="${EPISODE_TIME_S:-65}"
RESET_TIME_S="${RESET_TIME_S:-5}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# 4 相机 + 双臂 USB 负载高；若卡顿可 DISPLAY_DATA=false
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-0}"
IMG_WRITER_THREADS_PER_CAMERA="${IMG_WRITER_THREADS_PER_CAMERA:-1}"

if [ "${MODE}" = "record" ] && [ -z "${DATA_ROOT+x}" ]; then
    COLLECTION_DATE="${COLLECTION_DATE:-$(date +%Y%m%d_%H%M%S)}"
    DATA_ROOT="/home/rxn/datasets/bimanual_handover_4cam_${COLLECTION_DATE}"
    DATA_REPO="${DATA_REPO:-my_bimanual/handover_4cam_${COLLECTION_DATE}}"
else
    COLLECTION_DATE="${COLLECTION_DATE:-}"
    DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/bimanual_handover_4cam}"
    DATA_REPO="${DATA_REPO:-my_bimanual/handover_4cam}"
fi

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "  cd ${PROJECT_ROOT} && pip install -e \".[feetech]\""
    exit 1
fi
conda activate "${CONDA_ENV}"

echo "=============================="
echo "SO101 Bimanual Data Collection (${MODE})"
echo "=============================="
echo "Task:       ${SINGLE_TASK}"
if [ "${MODE}" = "record" ]; then
    echo "Date:       ${COLLECTION_DATE}"
    echo "Output:     ${DATA_ROOT}"
    echo "Episodes:   ${NUM_EPISODES}  (${EPISODE_TIME_S}s/ep, reset ${RESET_TIME_S}s)"
fi
echo "Cameras:    top_left=${TOP_LEFT_CAM}"
echo "            top_right=${TOP_RIGHT_CAM}"
echo "            left=${LEFT_CAM}"
echo "            right=${RIGHT_CAM}"
echo "Followers:  left=${LEFT_FOLLOWER_PORT}"
echo "            right=${RIGHT_FOLLOWER_PORT}"
echo "Leaders:    left=${LEFT_LEADER_PORT}"
echo "            right=${RIGHT_LEADER_PORT}"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
    echo "Stable symlinks (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || echo "  (none)"
    echo ""
    echo "Stable symlinks (/dev/serial/by-id):"
    ls -la /dev/serial/by-id/ 2>/dev/null || echo "  (none)"
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
    ("top", "${TOP_CAM}"),
    ("left", "${LEFT_CAM}"),
    ("right", "${RIGHT_CAM}"),
]
if not all(check(n, s) for n, s in cams):
    sys.exit(1)
PY

    echo ""
    echo "Rerun GUI：三合一 top | left | right，Ctrl+C 退出（无需机械臂）"
    echo ">>> 3 秒后开始..."
    sleep 3

    python - <<PY
import time
import cv2
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [
    ("top", "${TOP_CAM}"),
    ("left", "${LEFT_CAM}"),
    ("right", "${RIGHT_CAM}"),
]

def open_cam(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL: {name} ({src})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ${CAM_WIDTH})
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ${CAM_HEIGHT})
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(45):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK: {name} {frame.shape} ({src})")
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(f"FAIL: {name} no frame ({src})")

init_rerun(session_name="bimanual_preview")
caps = {name: open_cam(name, src) for name, src in CAMS}

def label_rgb(frame, text):
    out = frame.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out

print("Rerun 已开。主画面: observation.all (top | left | right 三合一)")
try:
    while True:
        t0 = time.perf_counter()
        obs = {}
        frames = []
        for name, cap in caps.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs[name] = rgb
                frames.append(label_rgb(rgb, name))
        if len(frames) == 3:
            obs["all"] = cv2.hconcat(frames)
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

if [ "${MODE}" = "preview4" ]; then
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
    echo "Rerun GUI：四合一 2x2 (top_left | top_right / left | right)，Ctrl+C 退出"
    echo ">>> 3 秒后开始..."
    sleep 3

    python - <<PY
import time
import cv2
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [
    ("top_left", "${TOP_LEFT_CAM}", ${TOP_LEFT_CAM_WIDTH}, ${TOP_LEFT_CAM_HEIGHT}),
    ("top_right", "${TOP_RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("left", "${LEFT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("right", "${RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
]

def open_cam(name, src, w, h):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL: {name} ({src})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(45):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK: {name} {frame.shape} ({src})")
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(f"FAIL: {name} no frame ({src})")

init_rerun(session_name="bimanual_preview4")
caps = {name: open_cam(name, src, w, h) for name, src, w, h in CAMS}
W, H = ${CAM_WIDTH}, ${CAM_HEIGHT}

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
    if [ ! -d "${DATA_ROOT}" ]; then
        echo "ERROR: dataset not found: ${DATA_ROOT}"
        exit 1
    fi
    lerobot-dataset-viz \
      --repo-id="${DATA_REPO}" \
      --root="${DATA_ROOT}" \
      --episode-index=0 \
      --tolerance-s=0.01 \
      --mode=local
    exit 0
fi

if [ "${MODE}" = "rerun" ]; then
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
    echo "Rerun 调相机：看 top_left / top_right / left / right，调好角度后 Ctrl+C 退出"
    echo ">>> 3 秒后开始..."
    sleep 3

    python - <<PY
import time
import rerun as rr
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

cfg = BiSO101FollowerConfig(
    id="${ROBOT_ID}",
    left_arm_port="${LEFT_FOLLOWER_PORT}",
    right_arm_port="${RIGHT_FOLLOWER_PORT}",
    cameras={
        "top_left": {"type": "opencv", "index_or_path": "${TOP_LEFT_CAM}", "width": ${TOP_LEFT_CAM_WIDTH}, "height": ${TOP_LEFT_CAM_HEIGHT}, "fps": ${FPS}, "fourcc": "MJPG", "warmup_s": ${CAM_WARMUP_S}},
        "top_right": {"type": "opencv", "index_or_path": "${TOP_RIGHT_CAM}", "width": ${CAM_WIDTH}, "height": ${CAM_HEIGHT}, "fps": ${FPS}, "fourcc": "MJPG", "warmup_s": ${CAM_WARMUP_S}},
        "left": {"type": "opencv", "index_or_path": "${LEFT_CAM}", "width": ${CAM_WIDTH}, "height": ${CAM_HEIGHT}, "fps": ${FPS}, "fourcc": "MJPG", "warmup_s": ${CAM_WARMUP_S}},
        "right": {"type": "opencv", "index_or_path": "${RIGHT_CAM}", "width": ${CAM_WIDTH}, "height": ${CAM_HEIGHT}, "fps": ${FPS}, "fourcc": "MJPG", "warmup_s": ${CAM_WARMUP_S}},
    },
)
init_rerun(session_name="bimanual_cam_tune")
robot = BiSO101Follower(cfg)
robot.connect()
print("Rerun 已开。在窗口里看 observation.top_left / top_right / left / right")
try:
    while True:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        log_rerun_data(observation=obs, action={})
        time.sleep(max(0, 1 / ${FPS} - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    rr.rerun_shutdown()
    robot.disconnect()
PY
    exit 0
fi

if [ "${MODE}" != "record" ]; then
    echo "Usage: bash get-data-bimanual.sh [record|cameras|preview|preview4|rerun|viz]"
    exit 1
fi

python -c "import scservo_sdk" 2>/dev/null || {
    echo "ERROR: scservo_sdk missing. pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
    exit 1
}

echo "Checking cameras..."
python - <<PY
import sys, cv2, time

def check(name, src, w=None, h=None):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        if w is not None and h is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(15):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                shape = frame.shape
                break
            time.sleep(0.1)
        else:
            ok = False
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({src})")
    return ok

failed = False
failed = not check("top_left", "${TOP_LEFT_CAM}", ${TOP_LEFT_CAM_WIDTH}, ${TOP_LEFT_CAM_HEIGHT}) or failed
failed = not check("top_right", "${TOP_RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}) or failed
failed = not check("left", "${LEFT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}) or failed
failed = not check("right", "${RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}) or failed
if failed:
    print("ERROR: camera check failed. Run: bash get-data-bimanual.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

RECORD_EPISODES="${NUM_EPISODES}"
RESUME_FLAG=""
if [ "${RESUME}" = "1" ]; then
    if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
        echo "ERROR: RESUME=1 but dataset not found: ${DATA_ROOT}"
        exit 1
    fi
    EXISTING_EPISODES="$(python3 -c "import json; print(json.load(open('${DATA_ROOT}/meta/info.json'))['total_episodes'])")"
    RECORD_EPISODES=$(( NUM_EPISODES - EXISTING_EPISODES ))
    if [ "${RECORD_EPISODES}" -le 0 ]; then
        echo "Already have ${EXISTING_EPISODES} episodes (target ${NUM_EPISODES}). Nothing to record."
        exit 0
    fi
    RESUME_FLAG="--resume=true"
    echo ""
    echo "Resume: ${EXISTING_EPISODES}/${NUM_EPISODES} done → recording ${RECORD_EPISODES} more"
fi

echo ""
echo "首次运行会依次校准 4 条臂（ACM0/ACM1 可复用原单臂校准，ACM2/ACM3 需完整校准）"
echo "快捷键: → 结束当前条 | ← 重录 | Esc 停止"
echo ">>> 3 秒后开始..."
sleep 3

lerobot-record \
  ${RESUME_FLAG} \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port="${LEFT_FOLLOWER_PORT}" \
  --robot.right_arm_port="${RIGHT_FOLLOWER_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port="${LEFT_LEADER_PORT}" \
  --teleop.right_arm_port="${RIGHT_LEADER_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --display_data="${DISPLAY_DATA}" \
  --play_sounds="${PLAY_SOUNDS}" \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.num_episodes="${RECORD_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --dataset.num_image_writer_threads_per_camera="${IMG_WRITER_THREADS_PER_CAMERA}"

echo "=============================="
echo "Recording finished"
echo "Dataset: ${DATA_ROOT}"
echo "Viz:     bash get-data-bimanual.sh viz"
echo "=============================="

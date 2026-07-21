source "$(dirname "$0")/assert_env_var.sh" || exit 1
RL_TASK_NAME="${RL_TASK_NAME:-pin_insertion}"
PREPARE_GPU="${GPU_RL_PREPARE:-auto}"
PASSTHROUGH_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --task)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --task requires a value" >&2
        exit 1
      fi
      RL_TASK_NAME="$2"
      shift 2
      ;;
    --task=*)
      RL_TASK_NAME="${1#--task=}"
      shift
      ;;
    --prepare-gpu)
      PREPARE_GPU="1"
      shift
      ;;
    --skip-prepare-gpu)
      PREPARE_GPU="0"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

CONFIG_FILE="$(dirname "$0")/tasks_config/${RL_TASK_NAME}/${RL_TASK_NAME}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: unknown RL task '${RL_TASK_NAME}' or missing config: ${CONFIG_FILE}" >&2
  exit 1
fi

if [ "${PREPARE_GPU}" = "auto" ]; then
  if [ "${RL_TASK_NAME}" = "gpu_insertion" ]; then
    PREPARE_GPU="1"
  else
    PREPARE_GPU="0"
  fi
fi

if [ "${RL_TASK_NAME}" = "gpu_insertion" ]; then
  export SAM3_SERVER_HOST="${SAM3_SERVER_HOST:-127.0.0.1}"
  export SAM3_SERVER_PORT="${SAM3_SERVER_PORT:-6767}"
fi

REQUIRE_SAM3="${GPU_RL_REQUIRE_SAM3:-auto}"
if [ "${REQUIRE_SAM3}" = "auto" ]; then
  if [ "${RL_TASK_NAME}" = "gpu_insertion" ]; then
    REQUIRE_SAM3="1"
  else
    REQUIRE_SAM3="0"
  fi
fi

if [ "${RL_TASK_NAME}" = "gpu_insertion" ] && [ "${REQUIRE_SAM3}" = "1" ]; then
  python3 - <<'PY'
import os
import sys
import urllib.error
import urllib.request

host = os.environ.get("SAM3_SERVER_HOST", "localhost")
port = int(os.environ.get("SAM3_SERVER_PORT", "9500"))
url = f"http://{host}:{port}/health"
try:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        if int(getattr(response, "status", 200)) >= 400:
            raise RuntimeError(f"HTTP {response.status}")
except (OSError, urllib.error.URLError, RuntimeError) as exc:
    print(
        f"ERROR: GPU insertion needs SAM3 for socket relocalization, but {url} "
        f"is not reachable ({exc}).",
        file=sys.stderr,
    )
    print(
        "Start it in the forge repo with:\n"
        f"  uv run python cap/skills/serve_sam3.py --port {port} --preload\n"
        "or export SAM3_SERVER_HOST/SAM3_SERVER_PORT for an existing server.\n"
        "Set GPU_RL_REQUIRE_SAM3=0 only if you intentionally disabled GPU slot relocalization.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
  if [ "$?" -ne 0 ]; then
    exit 1
  fi
fi

if [ "${RL_TASK_NAME}" = "gpu_insertion" ] && [ "${PREPARE_GPU}" = "1" ]; then
  echo "[rl_gear] Preparing GPU handover/socket hover before RL start..."
  GPU_PREP_LOG_DIR="${GPU_RL_PREP_LOG_DIR:-logs/gpu_handover_prepare_$(date +%Y%m%dT%H%M%S)}"
  GPU_HANDOVER_GO_HOME_ON_EXIT="${GPU_HANDOVER_GO_HOME_ON_EXIT:-0}" \
    GPU_REACTIVE_SOCKET_HOVER="${GPU_REACTIVE_SOCKET_HOVER:-0}" \
    GPU_SLOT_DEBUG_CAMERA="${GPU_RL_SLOT_HOVER_CAMERA:-top}" \
    GPU_SLOT_DEBUG_AUX_CAMERA="${GPU_RL_SLOT_HOVER_AUX_CAMERA:-left_third}" \
    GPU_SLOT_DEBUG_AUX_CAMERA_PREFER_WORLD_POSE="${GPU_RL_SLOT_HOVER_AUX_CAMERA_PREFER_WORLD_POSE:-1}" \
    GPU_SLOT_DEBUG_REQUIRE_AUX_CAMERA="${GPU_RL_SLOT_HOVER_REQUIRE_AUX_CAMERA:-0}" \
    GPU_SOCKET_HOVER_CAMERA="${GPU_RL_SLOT_HOVER_CAMERA:-top}" \
    GPU_SOCKET_HOVER_TRACK_CAMERA="${GPU_RL_SLOT_HOVER_CAMERA:-top}" \
    GPU_SOCKET_HOVER_AUX_CAMERA="${GPU_RL_SLOT_HOVER_AUX_CAMERA:-left_third}" \
    GPU_SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE="${GPU_RL_SLOT_HOVER_AUX_CAMERA_PREFER_WORLD_POSE:-1}" \
    GPU_SOCKET_HOVER_AUX_CAMERA_REQUIRED="${GPU_RL_SLOT_HOVER_REQUIRE_AUX_CAMERA:-0}" \
    GPU_SLOT_DEBUG_MAX_HOVER_XY_DELTA_M="${GPU_RL_SLOT_HOVER_MAX_XY_DELTA_M:-0.24}" \
    GPU_SLOT_DEBUG_AUX_CAMERA_MAX_POSE_DIFF_M="${GPU_RL_SLOT_HOVER_AUX_MAX_POSE_DIFF_M:-0.12}" \
    GPU_SOCKET_HOVER_AUX_MAX_POSE_DIFF_M="${GPU_RL_SLOT_HOVER_AUX_MAX_POSE_DIFF_M:-0.12}" \
    GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_AREA_PX="${GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_AREA_PX:-15000}" \
    GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_WIDTH_PX="${GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_WIDTH_PX:-110}" \
    GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_HEIGHT_PX="${GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_HEIGHT_PX:-100}" \
    uv run python run_script.py \
    robot=real_yam \
    script_file=cap/saved_scripts/gpu/gpu_handover.py \
    env.name=yam-real \
    skill_library_path=cap/saved_scripts/skill_library \
    script_output_dir="${GPU_PREP_LOG_DIR}" \
    robot.dashboard=false \
    robot.await_exit=false \
    robot.go_home_on_exit=false \
    runtime.exit_on_error=true
  gpu_prep_rc=$?
  gpu_prep_success="$(python3 - "${GPU_PREP_LOG_DIR}/result.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("0")
else:
    print("1" if bool(data.get("success", False)) else "0")
PY
)"
  if [ "${gpu_prep_success}" != "1" ]; then
    echo "ERROR: GPU handover/socket hover preparation failed; refusing to start RL." >&2
    echo "ERROR: preparation log: ${GPU_PREP_LOG_DIR}" >&2
    exit 1
  fi
  if [ "${gpu_prep_rc}" -ne 0 ]; then
    echo "[rl_gear] GPU preparation result is successful despite process exit ${gpu_prep_rc}; continuing."
  fi
fi

ROBOT_INTERFACE_PROFILE=1 uv run python rl/runner.py \
--config-file "${CONFIG_FILE}" \
--data-saving-path "$RL_DATA_PATH" \
"${PASSTHROUGH_ARGS[@]}"


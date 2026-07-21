#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/assert_env_var.sh"

RUN_STAMP="$(date +%Y%m%dT%H%M%S)"
REQUEST_DIR="${GPU_DUAL_FULL_CYCLE_REQUEST_DIR:-${RL_DATA_PATH}/gpu_insertion/full_cycle_requests}"
LOG_ROOT="${GPU_DUAL_FULL_CYCLE_LOG_ROOT:-logs/gpu_dual_full_cycle_${RUN_STAMP}}"
mkdir -p "${REQUEST_DIR}" "${LOG_ROOT}"

PASSTHROUGH_ARGS=("$@")

read_json_bool() {
  local path="$1"
  local key="$2"
  if [ ! -f "${path}" ]; then
    echo "missing"
    return 0
  fi
  python3 - "${path}" "${key}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("invalid")
else:
    value = data.get(sys.argv[2])
    if value is None and isinstance(data.get("details"), dict):
        value = data["details"].get(sys.argv[2])
    print("1" if bool(value) else "0")
PY
}

verify_initial_pick_source() {
  local handover_script="${REPO_ROOT}/cap/saved_scripts/gpu/gpu_handover.py"
  python3 - "${handover_script}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
stale_tokens = [
    "RIGHT_PICK_CLEAR_ON_START",
    "RIGHT_PICK_CLEAR_REQUIRED",
    "RIGHT_PICK_VALIDATE_TRAJECTORY",
    "RIGHT_PICK_APPROACH_RETRY_FLIPPED_YAW",
    "RIGHT_PICK_ROI_CANONICALIZE_YAW",
    "RIGHT_PICK_ROI_PREFER_FLIPPED_YAW",
    "PICK_RIGHT_ROI_USE_FULL_WIDTH",
    "PICK_RIGHT_ROI_EXCLUDE_RIGHT_EE",
    "PICK_RIGHT_ROI_RIGHT_EE_EXCLUSION_RADIUS_PX",
    "_right_pick_flipped_yaw_rpy",
    "prefer_flipped_yaw",
    "rpy_yaw_roi_flipped",
]
hits = [token for token in stale_tokens if token in text]
if hits:
    print(
        "ERROR: gpu_handover.py still contains stale initial-pick logic: "
        + ", ".join(hits),
        file=sys.stderr,
    )
    print(
        "ERROR: refusing to start the full-cycle loop because this can reproduce "
        "the RIGHT_PICK_CLEAR_ON_START crash or the ROI yaw flip.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

run_slot_insertion() {
  local slot="$1"
  local request_path="$2"
  local prep_log_dir="${LOG_ROOT}/slot${slot}_handover_prepare"
  local prepare_gpu="${GPU_DUAL_FULL_CYCLE_PREPARE:-${GPU_RL_PREPARE:-1}}"
  rm -f "${request_path}"
  rm -rf "${prep_log_dir}"
  echo "[gpu_dual_full_cycle] Slot ${slot}: look for a table GPU, prepare handover/socket hover, run RL insertion, wait for success..."
  echo "[gpu_dual_full_cycle] Slot ${slot}: request path: ${request_path}"
  if env \
    GPU_TARGET_SOCKET_NUMBER="${slot}" \
    GPU_RL_PREPARE="${prepare_gpu}" \
    GPU_RL_PREP_LOG_DIR="${prep_log_dir}" \
    GPU_HANDOVER_GO_HOME_ON_START="${GPU_DUAL_HANDOVER_GO_HOME_ON_START:-${GPU_HANDOVER_GO_HOME_ON_START:-1}}" \
    GPU_PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK="${GPU_PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK:-1}" \
    GPU_RIGHT_PICK_USE_FIXED_Z="${GPU_RIGHT_PICK_USE_FIXED_Z:-1}" \
    GPU_RIGHT_PICK_FIXED_Z_M="${GPU_RIGHT_PICK_FIXED_Z_M:-0.796}" \
    GPU_RIGHT_PICK_CANONICALIZE_YAW="${GPU_RIGHT_PICK_CANONICALIZE_YAW:-1}" \
    bash "${SCRIPT_DIR}/rl_gear.sh" \
    --task gpu_insertion \
    "${PASSTHROUGH_ARGS[@]}" \
    --gpu-success-full-cycle-enabled \
    --gpu-success-full-cycle-request-path "${request_path}"
  then
    local rl_rc=0
  else
    local rl_rc=$?
  fi

  if [ ! -f "${request_path}" ]; then
    local prep_success="skipped"
    local prep_left_holding_gpu="skipped"
    if [ "${prepare_gpu}" != "0" ]; then
      prep_success="$(read_json_bool "${prep_log_dir}/result.json" success)"
      prep_left_holding_gpu="$(read_json_bool "${prep_log_dir}/result.json" left_holding_gpu)"
    fi
    echo "[gpu_dual_full_cycle] Slot ${slot}: no success request was written; RL runner exit code=${rl_rc}." >&2
    echo "[gpu_dual_full_cycle] Slot ${slot}: handover prep log: ${prep_log_dir}" >&2
    if [ "${prepare_gpu}" != "0" ] && [ "${prep_success}" != "1" ]; then
      if [ "${prep_left_holding_gpu}" = "1" ]; then
        return 30
      fi
      if [ "${prep_success}" = "0" ]; then
        return 10
      fi
      return 40
    fi
    if [ "${rl_rc}" -ne 0 ]; then
      return "${rl_rc}"
    fi
    return 20
  fi
  echo "[gpu_dual_full_cycle] Slot ${slot}: success request detected."
}

run_press_only() {
  local slot="$1"
  local request_path="$2"
  local log_dir="${LOG_ROOT}/slot${slot}_press"
  echo "[gpu_dual_full_cycle] Slot ${slot}: press inserted GPU and go home; no unplug yet."
  set +e
  env \
    GPU_TARGET_SOCKET_NUMBER="${slot}" \
    GPU_FULL_CYCLE_REQUEST_PATH="${request_path}" \
    GPU_INSERT_PRESS_RUN_UNPLUG=0 \
    GPU_INSERT_PRESS_GO_HOME=1 \
    uv run python run_script.py \
    robot=real_yam \
    script_file=cap/saved_scripts/gpu/gpu_press_then_unplug.py \
    env.name=yam-real \
    skill_library_path=cap/saved_scripts/skill_library \
    script_output_dir="${log_dir}" \
    robot.dashboard=false \
    robot.await_exit=false \
    robot.go_home_on_exit=false \
    runtime.exit_on_error=true
  local press_rc=$?
  set -e
  if [ "${press_rc}" -ne 0 ]; then
    if [ -f "${log_dir}/result.json" ] && ! grep -q -- "--- error ---" "${log_dir}/exec.log" 2>/dev/null; then
      echo "[gpu_dual_full_cycle] Slot ${slot}: press script returned ${press_rc}, but no script error was logged; continuing."
    else
      echo "[gpu_dual_full_cycle] Slot ${slot}: press failed; log: ${log_dir}" >&2
      return "${press_rc}"
    fi
  fi
  echo "[gpu_dual_full_cycle] Slot ${slot}: press log: ${log_dir}"
}

reset_slot() {
  local slot="$1"
  local log_dir="${LOG_ROOT}/slot${slot}_reset"
  local drop_x drop_y drop_z drop_roll drop_pitch drop_yaw

  if [ "${slot}" = "1" ]; then
    drop_x="${GPU_DUAL_SLOT1_DROP_X:-${GPU_UNPLUG_LEFT_TABLE_PLACE_X:-0.528}}"
    drop_y="${GPU_DUAL_SLOT1_DROP_Y:-${GPU_UNPLUG_LEFT_TABLE_PLACE_Y:--0.154}}"
    drop_z="${GPU_DUAL_SLOT1_DROP_Z:-${GPU_UNPLUG_LEFT_TABLE_PLACE_Z:-0.864}}"
    drop_roll="${GPU_DUAL_SLOT1_DROP_ROLL_DEG:-${GPU_UNPLUG_LEFT_TABLE_PLACE_ROLL_DEG:--52.2}}"
    drop_pitch="${GPU_DUAL_SLOT1_DROP_PITCH_DEG:-${GPU_UNPLUG_LEFT_TABLE_PLACE_PITCH_DEG:-143.7}}"
    drop_yaw="${GPU_DUAL_SLOT1_DROP_YAW_DEG:-${GPU_UNPLUG_LEFT_TABLE_PLACE_YAW_DEG:-16.7}}"
  elif [ "${slot}" = "3" ]; then
    drop_x="${GPU_DUAL_SLOT3_DROP_X:-0.528}"
    drop_y="${GPU_DUAL_SLOT3_DROP_Y:--0.240}"
    drop_z="${GPU_DUAL_SLOT3_DROP_Z:-0.864}"
    drop_roll="${GPU_DUAL_SLOT3_DROP_ROLL_DEG:--52.2}"
    drop_pitch="${GPU_DUAL_SLOT3_DROP_PITCH_DEG:-143.7}"
    drop_yaw="${GPU_DUAL_SLOT3_DROP_YAW_DEG:-16.7}"
  else
    echo "[gpu_dual_full_cycle] unsupported reset slot: ${slot}" >&2
    return 1
  fi

  echo "[gpu_dual_full_cycle] Slot ${slot}: soft-bar unplug and place GPU at (${drop_x}, ${drop_y}, ${drop_z})."
  set +e
  env \
    GPU_TARGET_SOCKET_NUMBER="${slot}" \
    GPU_UNPLUG_GO_HOME_ON_START=1 \
    GPU_UNPLUG_LEFT_TABLE_PLACE_X="${drop_x}" \
    GPU_UNPLUG_LEFT_TABLE_PLACE_Y="${drop_y}" \
    GPU_UNPLUG_LEFT_TABLE_PLACE_Z="${drop_z}" \
    GPU_UNPLUG_LEFT_TABLE_PLACE_ROLL_DEG="${drop_roll}" \
    GPU_UNPLUG_LEFT_TABLE_PLACE_PITCH_DEG="${drop_pitch}" \
    GPU_UNPLUG_LEFT_TABLE_PLACE_YAW_DEG="${drop_yaw}" \
    uv run python run_script.py \
    robot=real_yam \
    script_file=cap/saved_scripts/gpu/gpu_reset_soft_bar.py \
    env.name=yam-real \
    skill_library_path=cap/saved_scripts/skill_library \
    script_output_dir="${log_dir}" \
    robot.dashboard=false \
    robot.await_exit=false \
    robot.go_home_on_exit=false \
    runtime.exit_on_error=true
  local reset_rc=$?
  set -e
  if [ "${reset_rc}" -ne 0 ]; then
    local reset_success
    reset_success="$(read_json_bool "${log_dir}/result.json" success)"
    if [ "${reset_success}" = "1" ]; then
      echo "[gpu_dual_full_cycle] Slot ${slot}: reset script returned ${reset_rc}, but result.json reports success; continuing."
    else
      echo "[gpu_dual_full_cycle] Slot ${slot}: reset failed; log: ${log_dir}" >&2
      return "${reset_rc}"
    fi
  fi
  echo "[gpu_dual_full_cycle] Slot ${slot}: reset log: ${log_dir}"
}

reset_both_slots() {
  local log_dir="${LOG_ROOT}/dual_reset"
  echo "[gpu_dual_full_cycle] Slots 3 then 1: soft-bar unplug both GPUs in one run_script process."
  set +e
  uv run python run_script.py \
    robot=real_yam \
    script_file=cap/saved_scripts/gpu/gpu_reset_soft_bar_dual.py \
    env.name=yam-real \
    skill_library_path=cap/saved_scripts/skill_library \
    script_output_dir="${log_dir}" \
    robot.dashboard=false \
    robot.await_exit=false \
    robot.go_home_on_exit=false \
    runtime.exit_on_error=true
  local reset_rc=$?
  set -e
  if [ "${reset_rc}" -ne 0 ]; then
    local reset_success
    reset_success="$(read_json_bool "${log_dir}/result.json" success)"
    if [ "${reset_success}" = "1" ]; then
      echo "[gpu_dual_full_cycle] Dual reset returned ${reset_rc}, but result.json reports success; continuing."
    else
      echo "[gpu_dual_full_cycle] Dual reset failed; log: ${log_dir}" >&2
      return "${reset_rc}"
    fi
  fi
  echo "[gpu_dual_full_cycle] Dual reset log: ${log_dir}"
}

slot1_request="${GPU_DUAL_SLOT1_REQUEST_PATH:-${REQUEST_DIR}/dual_${RUN_STAMP}_slot1_success.json}"
slot3_request="${GPU_DUAL_SLOT3_REQUEST_PATH:-${REQUEST_DIR}/dual_${RUN_STAMP}_slot3_success.json}"

echo "[gpu_dual_full_cycle] Starting two-GPU insertion full cycle."
echo "[gpu_dual_full_cycle] Log root: ${LOG_ROOT}"
echo "[gpu_dual_full_cycle] Sequence: slot 1 table-pick/insert/press, slot 3 table-pick/insert/press if another GPU is available, reset slot 3, reset slot 1."
verify_initial_pick_source

if [ "${GPU_DUAL_SKIP_SLOT1_INSERTION:-0}" = "1" ]; then
  if [ ! -f "${slot1_request}" ]; then
    echo "[gpu_dual_full_cycle] GPU_DUAL_SKIP_SLOT1_INSERTION=1 but slot-1 request is missing: ${slot1_request}" >&2
    exit 1
  fi
  echo "[gpu_dual_full_cycle] Slot 1: reusing existing success request: ${slot1_request}"
else
  run_slot_insertion 1 "${slot1_request}"
fi
if [ "${GPU_DUAL_SKIP_SLOT1_PRESS:-0}" = "1" ]; then
  echo "[gpu_dual_full_cycle] Slot 1: skipping press-only step; assuming it already completed."
else
  run_press_only 1 "${slot1_request}"
fi

echo "[gpu_dual_full_cycle] Checking for another GPU on the table through the slot-3 handover preparation."
set +e
run_slot_insertion 3 "${slot3_request}"
slot3_insert_rc=$?
set -e
if [ "${slot3_insert_rc}" -ne 0 ]; then
  if [ "${slot3_insert_rc}" -eq 10 ]; then
    echo "[gpu_dual_full_cycle] WARNING: no second GPU was successfully prepared for slot 3; assuming only one table GPU was available." >&2
    echo "[gpu_dual_full_cycle] WARNING: unplugging the already-inserted slot-1 GPU and ending the loop." >&2
    reset_slot 1
    echo "[gpu_dual_full_cycle] Complete with one GPU only."
    echo "[gpu_dual_full_cycle] Slot 1 request: ${slot1_request}"
    echo "[gpu_dual_full_cycle] Slot 3 request was not produced: ${slot3_request}"
    echo "[gpu_dual_full_cycle] Log root: ${LOG_ROOT}"
    exit 0
  fi
  if [ "${slot3_insert_rc}" -eq 30 ]; then
    echo "[gpu_dual_full_cycle] ERROR: slot-3 handover failed after the robot reported holding a GPU." >&2
    echo "[gpu_dual_full_cycle] ERROR: leaving automatic reset disabled to avoid moving with an unexpected held object." >&2
    echo "[gpu_dual_full_cycle] ERROR: inspect ${LOG_ROOT}/slot3_handover_prepare before recovering slot 1." >&2
    exit "${slot3_insert_rc}"
  fi
  if [ "${slot3_insert_rc}" -eq 40 ]; then
    echo "[gpu_dual_full_cycle] ERROR: slot-3 handover did not produce a readable result; not assuming the table is empty." >&2
    echo "[gpu_dual_full_cycle] ERROR: inspect ${LOG_ROOT}/slot3_handover_prepare before recovering slot 1." >&2
    exit "${slot3_insert_rc}"
  fi
  echo "[gpu_dual_full_cycle] Slot 3 insertion failed after preparation path rc=${slot3_insert_rc}. Resetting already-inserted slot 1 before exit." >&2
  reset_slot 1
  exit "${slot3_insert_rc}"
fi
run_press_only 3 "${slot3_request}"

reset_both_slots

echo "[gpu_dual_full_cycle] Complete."
echo "[gpu_dual_full_cycle] Slot 1 request: ${slot1_request}"
echo "[gpu_dual_full_cycle] Slot 3 request: ${slot3_request}"
echo "[gpu_dual_full_cycle] Log root: ${LOG_ROOT}"


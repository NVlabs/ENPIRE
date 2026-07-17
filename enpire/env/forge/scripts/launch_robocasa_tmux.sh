#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION="${SESSION:-robocasa_stack}"
ENV_NAME="${ENV_NAME:-robocasa:PrepareCoffee}"
SEED="${ROBOCASA_SEED:-42}"
CUROBO_PORT="${CAP_CUROBO_PORT:-9611}"
ENV_PORT="${CAP_PORT:-18600}"
SAM3_HOST="${SAM3_SERVER_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_SERVER_PORT:-26767}"
ANYGRASP_HOST="${ANYGRASP_SERVER_HOST:-127.0.0.1}"
ANYGRASP_PORT="${ANYGRASP_SERVER_PORT:-28122}"
ROBOT_TYPE="${CAP_ROBOT_TYPE:-panda}"
VISER_PORT="${VISER_PORT:-8890}"
ATTACH=1
KILL_EXISTING=1

usage() {
  cat <<EOF
Usage: $0 [options]

Launch RoboCasa EnvServer + Viser in a split tmux window.

Options:
  --task, --env NAME        RoboCasa env, e.g. robocasa:PrepareCoffee
  --seed N                 RoboCasa seed (default: $SEED)
  --session NAME           tmux session name (default: $SESSION)
  --curobo-port PORT       cuRobo port (default: $CUROBO_PORT)
  --env-port PORT          EnvServer / CAP_PORT (default: $ENV_PORT)
  --viser-port PORT        Viser UI port (default: $VISER_PORT)
  --sam3-host HOST         SAM3 server host (default: $SAM3_HOST)
  --sam3-port PORT         SAM3 server port (default: $SAM3_PORT)
  --anygrasp-host HOST     AnyGrasp server host (default: $ANYGRASP_HOST)
  --anygrasp-port PORT     AnyGrasp server port (default: $ANYGRASP_PORT)
  --no-attach              Start detached and print attach command
  --no-kill                Reuse/fail if tmux session already exists
  -h, --help               Show this help

Examples:
  $0 --task robocasa:PrepareCoffee
  $0 --task robocasa:KettleBoiling --seed 7 --session kettle
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task|--env)
      ENV_NAME="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --session)
      SESSION="$2"; shift 2 ;;
    --curobo-port)
      CUROBO_PORT="$2"; shift 2 ;;
    --env-port|--cap-port)
      ENV_PORT="$2"; shift 2 ;;
    --viser-port)
      VISER_PORT="$2"; shift 2 ;;
    --sam3-host)
      SAM3_HOST="$2"; shift 2 ;;
    --sam3-port)
      SAM3_PORT="$2"; shift 2 ;;
    --anygrasp-host)
      ANYGRASP_HOST="$2"; shift 2 ;;
    --anygrasp-port)
      ANYGRASP_PORT="$2"; shift 2 ;;
    --no-attach)
      ATTACH=0; shift ;;
    --no-kill)
      KILL_EXISTING=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$KILL_EXISTING" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session '$SESSION' already exists" >&2
    echo "Attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
fi

ENV_CMD="cd '$ROOT_DIR' && ROBOCASA_SEED='$SEED' CAP_CUROBO_PORT='$CUROBO_PORT' SAM3_SERVER_HOST='$SAM3_HOST' SAM3_SERVER_PORT='$SAM3_PORT' ANYGRASP_SERVER_HOST='$ANYGRASP_HOST' ANYGRASP_SERVER_PORT='$ANYGRASP_PORT' MUJOCO_GL=egl uv run python -m cap.env.robocasa.server --env '$ENV_NAME' --port '$ENV_PORT'"

VISER_CMD="cd '$ROOT_DIR' && CAP_PORT='$ENV_PORT' CAP_CUROBO_PORT='$CUROBO_PORT' SAM3_SERVER_HOST='$SAM3_HOST' SAM3_SERVER_PORT='$SAM3_PORT' ANYGRASP_SERVER_HOST='$ANYGRASP_HOST' ANYGRASP_SERVER_PORT='$ANYGRASP_PORT' uv run python tools/viser_curobo_planner.py --port '$VISER_PORT'"

tmux new-session -d -s "$SESSION" -n robocasa -c "$ROOT_DIR"
tmux send-keys -t "$SESSION:0.0" "$ENV_CMD" C-m
tmux split-window -h -t "$SESSION:0" -c "$ROOT_DIR"
tmux send-keys -t "$SESSION:0.1" "$VISER_CMD" C-m
tmux select-pane -t "$SESSION:0.0" -T "Env:$ENV_NAME"
tmux select-pane -t "$SESSION:0.1" -T "Viser:$VISER_PORT"
tmux select-layout -t "$SESSION:0" even-horizontal

cat <<EOF
Started tmux session: $SESSION
  left:  RoboCasa EnvServer $ENV_NAME on port $ENV_PORT
  right: Viser UI on port $VISER_PORT

External dependencies:
  cuRobo already running on port $CUROBO_PORT
  SAM3    @ $SAM3_HOST:$SAM3_PORT
  AnyGrasp @ $ANYGRASP_HOST:$ANYGRASP_PORT

Attach:
  tmux attach -t $SESSION

Open Viser:
  http://127.0.0.1:$VISER_PORT
EOF

if [[ "$ATTACH" == "1" ]]; then
  tmux attach -t "$SESSION"
fi

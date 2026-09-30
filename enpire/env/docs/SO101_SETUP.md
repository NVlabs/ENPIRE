# SO-101 setup

ENPIRE can drive a single SO-101 arm through LeRobot's Feetech driver. This
covers joint-level control, forward and inverse kinematics, the CaP robot
profile, and leader-arm teleoperation. Cameras, grasp planning, cuRobo, and
the real-world RL stack still assume a YAM station and are not ported.

## What differs from YAM

| | YAM | SO-101 |
|---|---|---|
| Arm joints | 6 | 5 |
| Motors | DaMiao over CAN, impedance control | Feetech STS3215 over USB serial, position control |
| Gains | `kp`/`kd` per command | Ignored. The servos run their own position loop |
| Force estimate, gravity comp | Yes | No |
| Arms per station | 2 | 1 (named `left` by default) |

With five joints, most 6D pose targets are not reachable. `So101Kinematics`
weights position above orientation and returns the residual errors, so check
them before commanding a solution.

## Install

```bash
uv sync --extra dev --extra control-so101
```

`control-so101` pins LeRobot 0.4.x because LeRobot 0.5 needs Python 3.12. It
cannot be combined with `vision-local`.

Your user needs read and write access to the serial device, usually through
the `dialout` group.

## Calibrate with LeRobot

ENPIRE never runs LeRobot's interactive calibration. Find the ports and
calibrate each arm once with the LeRobot tools. Calibration asks you to move
the arm by hand.

```bash
uv run lerobot-find-port
uv run lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 --robot.id=enpire_so101_follower
uv run lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 --teleop.id=enpire_so101_leader
```

The ids must match the ones the arm server uses. Defaults come from these
environment variables:

| Variable | Default |
|---|---|
| `ENPIRE_SO101_FOLLOWER_SERIAL` | `/dev/ttyACM0` |
| `ENPIRE_SO101_FOLLOWER_ID` | `enpire_so101_follower` |
| `ENPIRE_SO101_LEADER_SERIAL` | `/dev/ttyACM1` |
| `ENPIRE_SO101_LEADER_ID` | `enpire_so101_leader` |

## Run the arm server

Check the resolved configuration first:

```bash
uv run python -m enpire.env.forge.robot.so101.arm_server --dry-run
```

Then start the follower. `--max-relative-target` caps each command's joint
change in degrees and is a sensible guard for first runs.

```bash
uv run python -m enpire.env.forge.robot.so101.arm_server --max-relative-target 5
```

Use `--mode leader` to serve the leader arm on port 11338. The follower uses
port 11337 and exposes the same RPCs as the YAM follower server, so
`FollowerRobotClient` works against it.

## Use it from Python

```python
from enpire.env.forge.cap.env.real_so101 import RealSo101Env

env = RealSo101Env()
obs = env.get_observations("left")
env.go_home()
env.set_gripper("left", 0.0)
```

Joint values are radians with zero at the middle of each joint range, which
is the LeRobot degrees calibration. The gripper is normalized to `[0, 1]` with
`0` closed. Commands outside the joint limits are clipped by the driver.

## Model

`enpire/env/forge/robot/models/so101/so101.xml` is the kinematic tree of the
SO-ARM100 `so101_new_calib` model with meshes removed. See
`source_provenance.yaml` for the pinned upstream commit.

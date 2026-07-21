# Record and Replay Trajectory Example

Record and replay robot arm movements using the I2RT Python API.

## Quick Start

```bash
python record_replay_trajectory.py
```

## What It Does

- **Record**: Move the robot arm by hand while it captures joint positions at 30Hz
- **Replay**: Robot automatically reproduces your recorded motion
- **Save**: Store trajectories as `.npy` files for later use
- **Load**: Load previously saved trajectories from files

## Controls

- `r` - Start/stop recording
- `p` - Play back recorded motion
- `s` - Save trajectory to file
- `l` - Load trajectory from file
- `q` - Quit

## What You'll See

```
Controls:
  r : Start/stop recording
  p : Start replay
  s : Save trajectory
  l : Load trajectory from file
  q : Quit

Status:
Recording: False  Replaying: False
Trajectory length: 0 samples
Press 'q' to quit.
```

## Workflow

1. Run the script
2. **Option A**: Press `r` to start recording, move arm, press `r` to stop
3. **Option B**: Press `l` to load a previously saved trajectory file
4. Press `p` to replay the motion
5. Press `s` to save (optional)
6. Press `q` to quit

## Options

```bash
--channel can0          # CAN bus channel
--gripper linear_4310   # Yam gripper type so the gravity compensation can load the correct model
--output file.npy       # Output filename
--load file.npy         # Load trajectory at startup
```

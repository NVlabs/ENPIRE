# Third-Party Notices

ENPIRE is Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

This file provides third-party notices required by components used in ENPIRE.
Each third-party component retains its original license; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full license texts and links.

---

## Vendored source code (`third_party/`)

The following projects are vendored as editable local packages under `third_party/` and are not uploaded to PyPI by this repository.

| Component | Path | Origin | License |
|-----------|------|--------|---------|
| cuRobo | `third_party/curobo/` | [NVlabs/curobo](https://github.com/NVlabs/curobo) | Apache-2.0 |
| PyRoki | `third_party/pyroki/` | [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | MIT — Copyright (c) 2025 Chung Min Kim |
| i2rt | `third_party/i2rt/` | [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | MIT — Copyright (c) I2RT Robotics |
| RealtimeSTT | `third_party/ui/RealtimeSTT/` | [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | MIT — Copyright (c) 2023 Kolja Beigel |
| AgentLace | `enpire/policy/pld/runtime/vendor/agentlace/` | [youliangtan/agentlace](https://github.com/youliangtan/agentlace) | MIT — Copyright (c) 2023 You Liang Tan |
| SERL-launcher | `enpire/policy/pld/runtime/vendor/serl_launcher/` | [rail-berkeley/serl](https://github.com/rail-berkeley/serl) | MIT — Copyright (c) Meta Platforms, Inc. and affiliates. |

## NVIDIA-authored components under `third_party/`

The following component is authored by NVIDIA and vendored locally for packaging convenience. It is distributed under the same Apache-2.0 license as this repository.

| Component | Path | License |
|-----------|------|---------|
| overlay_viz | `third_party/overlay_viz/` | Apache-2.0 (NVIDIA CORPORATION) |

## Python package dependencies

This project installs Python packages from PyPI at build time.  Optional extras add additional dependencies; see `pyproject.toml` for per-extra lists.

### Core

- numpy (BSD-3-Clause)
- PyYAML (MIT)

### Vision

- opencv-python / opencv-contrib-python (Apache-2.0)
- Pillow (HPND)
- requests (Apache-2.0)
- PyTorch (BSD-3-Clause)
- torchvision (BSD-3-Clause)
- Transformers (Apache-2.0)
- FastAPI (MIT)
- uvicorn (BSD-3-Clause)

### Grasping

- GraspNetAPI (MIT)
- Open3D (MIT)

### Planning and kinematics

- MuJoCo (Apache-2.0)
- Mink (Apache-2.0)
- Pink / Pinocchio (BSD-2-Clause)
- Portal (Apache-2.0)
- PyRoki — vendored, see above
- cuRobo — vendored, see above
- SciPy (BSD-3-Clause)
- yourdfpy (MIT)
- NVIDIA Warp (Apache-2.0)
- ninja (Apache-2.0)

### Robot control

- damiao-motor (MIT)
- gymnasium (MIT)
- msgpack (Apache-2.0)
- pyserial (BSD-3-Clause)
- tyro (MIT)
- pyzmq (BSD-3-Clause)
- i2rt — vendored, see above

### Cameras

- pyrealsense2 — Intel RealSense SDK Python bindings (Apache-2.0)
- pyzed — Stereolabs ZED Python API (Stereolabs SDK License)

### Calibration

- opencv-contrib-python (Apache-2.0)
- damiao-motor (MIT)
- MuJoCo (Apache-2.0)
- Portal (Apache-2.0)
- pyrealsense2 (Apache-2.0)
- SciPy (BSD-3-Clause)

### VLM

- google-genai (Apache-2.0)
- openai (Apache-2.0)

### Code-as-Policy (CaP)

- hydra-core (MIT)
- omegaconf (Apache-2.0)

### Real-world RL

- evdev (MIT)
- lz4 (BSD-2-Clause)
- pynput (LGPLv3)
- FastAPI (MIT)
- PyTorch (BSD-3-Clause)
- Portal (Apache-2.0)

### PLD actor / learner (isolated runtime under `enpire/policy/pld/runtime/`)

- JAX (Apache-2.0)
- AgentLace — vendored, see above
- SERL-launcher — vendored, see above
- protobuf (BSD-3-Clause)
- rich (MIT)
- matplotlib (PSF / BSD-compatible)

### Simulation

- numba (BSD-2-Clause)
- hidapi (BSD-3-Clause)
- qpsolvers (LGPLv3)

---

For full license texts, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

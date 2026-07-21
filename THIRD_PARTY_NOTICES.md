# Third-Party Notices

ENPIRE is a practitioner-oriented harness for repeatable robot policy improvement built on top of the Forge runtime and YAM calibration pipeline.  All third-party software used by this project retains its original license; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full license texts and links.

> **License status notice:** The root release license and IP review for source code migrated from the internal Forge branches and yam-calibration repository are still pending. Third-party components listed below retain their own licenses independently of that review. Do not redistribute this software publicly until the root license is resolved.

---

## Vendored source code (`third_party/`)

The following projects are vendored as editable local packages under `third_party/` and are not uploaded to PyPI by this repository.

| Component | Origin | License |
|-----------|--------|---------|
| cuRobo | [NVlabs/curobo](https://github.com/NVlabs/curobo) | NVIDIA Research License |
| PyRoki | [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | MIT |
| i2rt | [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | MIT |
| RoboCasa | [robocasa/robocasa](https://github.com/robocasa/robocasa) | MIT |
| robosuite | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT |

## Source-derived code

The following Forge feature branches and the yam-calibration repository are the upstream source of code incorporated into `enpire/env/forge/` and `cap/`.  Their license status is tracked in `enpire/env/docs/source_provenance.yaml`.

| Component | Branch / Commit | License Status |
|-----------|----------------|---------------|
| Forge – GPU insertion | `haotian/gpu-insertion @ 682f7937` | Pending source-owner review |
| Forge – Zip-tie AutoRL | `tonghe/ziptie-autorl @ 1abbfeae` | Pending source-owner review |
| Forge – PushT | `wenlix/pusht_env @ 3cc5e899` | Pending source-owner review |
| Forge – Pin / AutoRL | `wenlix/autorl @ 4c37817d` | Pending source-owner review |
| yam-calibration | `main @ 37babca` | Pending source-owner review |

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
- AgentLace (MIT)
- SERL-launcher (MIT)
- protobuf (BSD-3-Clause)
- rich (MIT)
- matplotlib (PSF / BSD-compatible)

### Simulation

- RoboCasa — vendored, see above
- robosuite — vendored, see above
- numba (BSD-2-Clause)
- hidapi (BSD-3-Clause)
- qpsolvers (LGPLv3)

---

For full license texts, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

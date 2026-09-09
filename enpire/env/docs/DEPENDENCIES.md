# ENPIRE dependencies

ENPIRE uses Python 3.11 and [uv](https://github.com/astral-sh/uv). Exact
versions are defined by [`pyproject.toml`](../../../pyproject.toml) and
[`uv.lock`](../../../uv.lock). The PLD learner has an isolated environment at
[`enpire/policy/pld/runtime`](../../policy/pld/runtime/pyproject.toml).

## Install

See [`INSTALL.md`](INSTALL.md) — it is the single source for the clone-with-
submodules step and the extras matrix. This file covers what those dependencies
*are*, not how to install them.

## Major third-party projects

| Dependency | Used for | Install/source | Upstream |
|---|---|---|---|
| cuRobo v0.8.0 | GPU IK, collision checking, and trajectory generation | `planning-local`; Apache-2.0 submodule in `third_party/curobo` | [NVlabs/curobo](https://github.com/NVlabs/curobo) |
| CUDA Python (`cuda.core`) | Runtime compilation and launch of cuRobo v0.8 CUDA kernels | `planning-local`; selected through `nvidia-curobo[cu12]` | [NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python) |
| NVIDIA Warp | cuRobo geometry and perception utilities | `planning-local`; pinned to `1.12.0` | [NVIDIA/warp](https://github.com/NVIDIA/warp) |
| MuJoCo | YAM simulation, models, gravity compensation, and IK | `planning`, `control-yam` | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) |
| Mink | MuJoCo differential IK | `planning`, `control-yam` | [kevinzakka/mink](https://github.com/kevinzakka/mink) |
| Pink / Pinocchio | Alternate rigid-body kinematics | `planning` | [stephane-caron/pink](https://github.com/stephane-caron/pink) |
| PyRoki | JAX kinematic optimization | `planning`; vendored in `third_party/pyroki` | [chungmin99/pyroki](https://github.com/chungmin99/pyroki) |
| I2RT | YAM leader/handle control and gravity models | `control-i2rt`; vendored in `third_party/i2rt` | [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) |
| damiao-motor | DaMiao motor and SocketCAN control | `control-yam` | [jia-xie/python-damiao-driver](https://github.com/jia-xie/python-damiao-driver) |
| SAM3 / Transformers | Text-prompted object segmentation | `vision-local`; weights external | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| AnyGrasp / GraspNetAPI | Learned 6-DoF grasp proposals | `grasping-local`; SDK and license external | [graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk) |
| BundleSDF | Optional unknown-object pose tracking | External runtime; ENPIRE supplies adapters | [NVlabs/BundleSDF](https://github.com/NVlabs/BundleSDF) |
| RealSense SDK | Intel RGB-D cameras | `camera-realsense` | [realsenseai/librealsense](https://github.com/realsenseai/librealsense) |
| ZED SDK | Optional ZED top camera | `camera-zed`; host SDK required | [stereolabs/zed-sdk](https://github.com/stereolabs/zed-sdk) |
| SERL / HIL-SERL / AgentLace | PLD actor, learner, replay, SAC/RLPD, and distributed transport | Vendored in the isolated PLD runtime | [HIL-SERL](https://github.com/rail-berkeley/hil-serl), [AgentLace](https://github.com/youliangtan/agentlace) |
| JAX / Flax / Optax | PLD networks and optimization | Isolated PLD environment | [jax-ml/jax](https://github.com/jax-ml/jax) |
| OpenAI / Google GenAI | Optional hosted VLM backends | `vlm`; credentials external | [openai/openai-python](https://github.com/openai/openai-python), [googleapis/python-genai](https://github.com/googleapis/python-genai) |

Routine Python libraries are intentionally omitted here; see the manifests for
the complete locked package graph.

## Required system software

- Linux x86-64, Git, Git LFS, tmux, FFmpeg, curl, and uv.
- SocketCAN/udev support for YAM hardware.
- NVIDIA driver for cuRobo, local vision, AnyGrasp, or PLD. A *system* CUDA
  toolkit is not required for cuRobo — `nvidia-curobo[cu12]` installs its own
  toolchain as wheels (see [`CUROBO_SETUP.md`](CUROBO_SETUP.md)). Building the
  AnyGrasp SDK yourself does need a local toolkit.
- Librealsense-compatible host support or the ZED SDK for those cameras.

## Important compatibility pins

- Python `>=3.11,<3.12`
- Warp `==1.12.0` for the vendored cuRobo API
- RealSense Python `==2.56.5.9235`
- Mink `==0.0.12`; yourdfpy `==0.0.59`
- PLD JAX/JAXlib `==0.6.1`, NumPy `<2`, protobuf `==5.29.6`

## Not stored in Git

API keys, W&B/Hugging Face credentials, datasets, checkpoints, SAM3 weights,
the licensed AnyGrasp SDK/checkpoint/license, BundleSDF weights, generated udev
rules, and station-specific calibrated XML/URDF files remain external. Reusable
YAM/Fello models and redistributable source code are included in this repository.

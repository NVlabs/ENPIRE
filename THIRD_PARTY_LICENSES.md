# Third-Party Licenses

This document lists all third-party software incorporated into or distributed with ENPIRE, together with their licenses.  Vendored projects whose full license text differs from a standard SPDX identifier are reproduced below the table.

## Summary table

| No. | Package | Category | License | Source / License Link |
|-----|---------|----------|---------|----------------------|
| 1 | cuRobo | git submodule @ v0.7.7 (planning) | NVIDIA Research License | [NVlabs/curobo – LICENSE](https://github.com/NVlabs/curobo/blob/main/LICENSE) |
| 2 | PyRoki | vendored (planning) | MIT | [chungmin99/pyroki – LICENSE](https://github.com/chungmin99/pyroki/blob/main/LICENSE) |
| 3 | i2rt | vendored (control) | MIT | [i2rt-robotics/i2rt – LICENSE](https://github.com/i2rt-robotics/i2rt/blob/main/LICENSE) |
| 4 | RealtimeSTT | vendored (voice) | MIT | [KoljaB/RealtimeSTT – LICENSE](https://github.com/KoljaB/RealtimeSTT/blob/master/LICENSE) |
| 5 | numpy | core | BSD-3-Clause | [numpy/numpy – LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| 6 | PyYAML | core | MIT | [yaml/pyyaml – LICENSE](https://github.com/yaml/pyyaml/blob/master/LICENSE) |
| 7 | opencv-python / opencv-contrib-python | vision, calibration | Apache-2.0 | [opencv/opencv – LICENSE](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| 9 | Pillow | vision, VLM | HPND | [python-pillow/Pillow – LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| 10 | requests | vision, VLM | Apache-2.0 | [psf/requests – LICENSE](https://github.com/psf/requests/blob/main/LICENSE) |
| 11 | PyTorch | vision-local, real-RL | BSD-3-Clause | [pytorch/pytorch – LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| 12 | torchvision | vision-local | BSD-3-Clause | [pytorch/vision – LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) |
| 13 | Transformers (HuggingFace) | vision-local | Apache-2.0 | [huggingface/transformers – LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| 14 | FastAPI | vision-local, grasping, real-RL | MIT | [fastapi/fastapi – LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| 15 | uvicorn | vision-local, grasping, real-RL | BSD-3-Clause | [encode/uvicorn – LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.md) |
| 16 | GraspNetAPI | grasping-local | MIT | [graspnet/graspnetAPI – LICENSE](https://github.com/graspnet/graspnetAPI/blob/master/LICENSE) |
| 17 | Open3D | grasping-local | MIT | [isl-org/Open3D – LICENSE](https://github.com/isl-org/Open3D/blob/main/LICENSE) |
| 18 | MuJoCo | planning, control-yam, calibration | Apache-2.0 | [google-deepmind/mujoco – LICENSE](https://github.com/google-deepmind/mujoco/blob/main/LICENSE) |
| 19 | Mink | planning, control-yam | Apache-2.0 | [kevinzakka/mink – LICENSE](https://github.com/kevinzakka/mink/blob/main/LICENSE) |
| 20 | Pink / Pinocchio | planning | BSD-2-Clause | [stephane-caron/pink – LICENSE](https://github.com/stephane-caron/pink/blob/main/LICENSE) |
| 21 | Portal | planning, control-yam, calibration, real-RL | Apache-2.0 | [danijar/portal – LICENSE](https://github.com/danijar/portal/blob/main/LICENSE) |
| 22 | SciPy | planning, calibration, real-RL | BSD-3-Clause | [scipy/scipy – LICENSE](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| 23 | yourdfpy | planning | MIT | [clemense/yourdfpy – LICENSE](https://github.com/clemense/yourdfpy/blob/main/LICENSE) |
| 24 | NVIDIA Warp | planning-local | Apache-2.0 | [NVIDIA/warp – LICENSE](https://github.com/NVIDIA/warp/blob/main/LICENSE.md) |
| 25 | ninja | planning-local | Apache-2.0 | [ninja-build/ninja – COPYING](https://github.com/ninja-build/ninja/blob/master/COPYING) |
| 26 | damiao-motor | control-yam, calibration | MIT | [jia-xie/python-damiao-driver – LICENSE](https://github.com/jia-xie/python-damiao-driver/blob/main/LICENSE) |
| 27 | gymnasium | control-yam, simulation | MIT | [Farama-Foundation/Gymnasium – LICENSE](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) |
| 28 | msgpack | control-yam | Apache-2.0 | [msgpack/msgpack-python – COPYING](https://github.com/msgpack/msgpack-python/blob/main/COPYING) |
| 29 | pyserial | control-yam | BSD-3-Clause | [pyserial/pyserial – LICENSE](https://github.com/pyserial/pyserial/blob/master/LICENSE.txt) |
| 30 | tyro | control-yam, real-RL | MIT | [brentyi/tyro – LICENSE](https://github.com/brentyi/tyro/blob/main/LICENSE) |
| 31 | pyzmq | control-yam, real-RL | BSD-3-Clause | [zeromq/pyzmq – LICENSE](https://github.com/zeromq/pyzmq/blob/main/LICENSE.md) |
| 32 | pyrealsense2 | camera-realsense, calibration | Apache-2.0 | [IntelRealSense/librealsense – LICENSE](https://github.com/IntelRealSense/librealsense/blob/master/LICENSE) |
| 33 | pyzed (Stereolabs ZED SDK) | camera-zed | Stereolabs SDK License | [stereolabs/zed-python-api](https://github.com/stereolabs/zed-python-api) |
| 34 | google-genai | vlm | Apache-2.0 | [googleapis/python-genai – LICENSE](https://github.com/googleapis/python-genai/blob/main/LICENSE) |
| 35 | openai | vlm | Apache-2.0 | [openai/openai-python – LICENSE](https://github.com/openai/openai-python/blob/main/LICENSE) |
| 36 | hydra-core | cap, pld | MIT | [facebookresearch/hydra – LICENSE](https://github.com/facebookresearch/hydra/blob/main/LICENSE) |
| 37 | omegaconf | cap, pld | Apache-2.0 | [omry/omegaconf – LICENSE](https://github.com/omry/omegaconf/blob/master/LICENSE) |
| 38 | evdev | real-RL | MIT | [gvalkov/python-evdev – LICENSE](https://github.com/gvalkov/python-evdev/blob/main/LICENSE) |
| 39 | lz4 | real-RL | BSD-2-Clause | [python-lz4/python-lz4 – LICENSE](https://github.com/python-lz4/python-lz4/blob/master/LICENSE) |
| 40 | pynput | real-RL | LGPLv3 | [moses-palmer/pynput – COPYING](https://github.com/moses-palmer/pynput/blob/master/COPYING) |
| 41 | JAX | pld-runtime | Apache-2.0 | [jax-ml/jax – LICENSE](https://github.com/jax-ml/jax/blob/main/LICENSE) |
| 42 | AgentLace | pld-runtime | MIT | [youliangtan/agentlace – LICENSE](https://github.com/youliangtan/agentlace/blob/main/LICENSE) |
| 43 | SERL / HIL-SERL (serl-launcher) | pld-runtime | MIT | [rail-berkeley/hil-serl – LICENSE](https://github.com/rail-berkeley/hil-serl/blob/main/LICENSE) |
| 44 | protobuf | pld-runtime | BSD-3-Clause | [protocolbuffers/protobuf – LICENSE](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE) |
| 45 | rich | pld-runtime | MIT | [Textualize/rich – LICENSE](https://github.com/Textualize/rich/blob/master/LICENSE) |
| 46 | matplotlib | pld-runtime | PSF / BSD-compatible | [matplotlib/matplotlib – LICENSE](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |

---

## Full license texts for vendored components

### 1. cuRobo — NVIDIA Research License

cuRobo is included as a git submodule at `third_party/curobo/` (tag v0.7.7, commit `0a50de1`).

```
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA License

1. Definitions

"Licensor" means any person or entity that distributes its Work.
"Work" means (a) the original work of authorship made available under this
license, which may include software, documentation, or other files, and (b)
any additions to or derivative works thereof that are made available under
this license.

2. License Grant

2.1 Copyright Grant. Subject to the terms and conditions of this license,
each Licensor grants to you a perpetual, worldwide, non-exclusive,
royalty-free, copyright license to use, reproduce, prepare derivative works
of, publicly display, publicly perform, sublicense and distribute its Work
and any resulting derivative works in any form.

3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you
do so under this License, (b) you include a complete copy of this License
with your distribution, and (c) you retain without modification any copyright,
patent, trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply
to the use, reproduction, and distribution of your derivative works of the
Work ("Your Terms") only if (a) Your Terms provide that the use limitation in
Section 3.3 applies to your derivative works, and (b) you identify the
specific derivative works that are subject to Your Terms. Notwithstanding Your
Terms, this License (including the redistribution requirements in Section 3.1)
will continue to apply to the Work itself.

3.3 Use Limitation. The Work and any derivative works thereof only may be
used or intended for use with the web services, computing platforms or
applications provided by NVIDIA Corporation or its affiliates (collectively
"NVIDIA Platforms").

3.4 Patent Claims. If you bring or threaten to bring a patent claim against
any Licensor (including any claim, cross-claim or counterclaim in a lawsuit)
to enforce any patents that you allege are infringed by any Work, then your
rights under this License from such Licensor (including the grant in Section
2.1) will terminate immediately.

3.5 Trademarks. You are not granted any right or license under this License
to use any trademarks, service marks, trade names, products, or services of
NVIDIA Corporation. You may use the NVIDIA name solely to identify the origin
of Work subject to the requirements of this License.

3.6 Termination. If you violate any term of this License, then your rights
under this License (including the grant in Section 2.1) will terminate
immediately.

4. Disclaimer of Warranty

THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTIES OR
CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE
APPROPRIATENESS OF USING OR REDISTRIBUTING THE WORK AND ASSUME ANY RISKS
ASSOCIATED WITH YOUR EXERCISE OF PERMISSIONS UNDER THIS LICENSE.

5. Limitation of Liability

IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE),
CONTRACT, OR OTHERWISE, UNLESS REQUIRED BY APPLICABLE LAW (SUCH AS
DELIBERATE AND GROSSLY NEGLIGENT ACTS) OR AGREED TO IN WRITING, SHALL ANY
LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A
RESULT OF THIS LICENSE OR OUT OF THE USE OR INABILITY TO USE THE WORK,
EVEN IF SUCH LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

---

### 2. PyRoki — MIT License

PyRoki is vendored under `third_party/pyroki/`.

```
MIT License

Copyright (c) 2025 Chung Min Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

### 3. i2rt — MIT License

i2rt is vendored under `third_party/i2rt/`.

```
MIT License

Copyright (c) I2RT Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Notes on specific packages

**cuRobo (NVIDIA Research License):** The NVIDIA Research License restricts use to NVIDIA Platforms and is **not** a permissive open-source license.  cuRobo must not be used for commercial applications outside of NVIDIA services without a separate agreement.  The `planning-local` extra that installs cuRobo is opt-in and is not required for any other ENPIRE capability.

**pyzed / Stereolabs ZED SDK:** The ZED Python API requires acceptance of the Stereolabs SDK License Agreement.  The `camera-zed` extra is opt-in.  The ZED host SDK must be installed separately from Stereolabs.

**pynput (LGPLv3):** pynput is installed as a shared library linked at runtime and is not statically compiled into ENPIRE.  Its use under LGPLv3 does not impose copyleft obligations on ENPIRE application code provided the package is not modified.

**AnyGrasp SDK:** The licensed AnyGrasp SDK, checkpoint, and license archive are **not** included in this repository and must be obtained directly from GraspNet.  Only the open-source GraspNetAPI is listed above.

**damiao-motor:** Installed from the local path `../python-damiao-driver` during development; the upstream project is MIT-licensed.

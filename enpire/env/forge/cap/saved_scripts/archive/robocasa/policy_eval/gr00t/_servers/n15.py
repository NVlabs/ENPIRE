# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ZMQ server for GR00T N1.5 model.

The benchmark repo embeds the server inside run_eval.py. This script
extracts it so we can launch one server per GPU, matching our multi-GPU
architecture.

Launched by run_eval_365.py as a subprocess — not run directly.
"""

import argparse

# The benchmark repo must be on PYTHONPATH for gr00t imports.

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def main():
    p = argparse.ArgumentParser(description="GR00T N1.5 inference server")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-config", default="panda_omron")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--denoising-steps", type=int, default=4)
    args = p.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    print(f"Loading N1.5 model from {args.model_path}...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    print(f"Model loaded. Starting server on port {args.port}...")

    server = RobotInferenceServer(policy, port=args.port)
    server.run()


if __name__ == "__main__":
    main()

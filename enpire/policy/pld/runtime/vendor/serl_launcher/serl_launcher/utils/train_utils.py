# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT

import os
import pickle as pkl
from collections import defaultdict

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
import wandb
from flax.core import frozen_dict
from tqdm import tqdm


def ask_for_frame(images_dict):
    # Create a new figure
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i, (idx, img) in enumerate(images_dict.items()):
        # Display the image
        axes[i].imshow(img)

        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Overlay the index number
        axes[i].text(
            10,
            30,
            str(idx),
            color="white",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.show(block=False)

    while True:
        try:
            first_success = int(input("First success frame number: "))
            assert first_success in images_dict.keys()
            break
        except:
            continue

    plt.close(fig)

    return first_success


def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if not isinstance(offline_batch, dict):
        offline_batch = offline_batch.unfreeze()

    if not isinstance(online_batch, dict):
        online_batch = online_batch.unfreeze()

    for k, v in offline_batch.items():
        if type(v) is dict:
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            batch[k] = jnp.concatenate((offline_batch[k], online_batch[k]), axis=axis)

    return frozen_dict.freeze(batch)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def _replace_resnet_encoder_params(agent, image_keys, encoder_params, label):
    new_params = agent.state.params

    for module_key in ("modules_actor", "modules_critic"):
        if module_key not in new_params:
            continue
        if "encoder" not in new_params[module_key]:
            continue
        for image_key in image_keys:
            new_encoder_params = new_params[module_key]["encoder"][f"encoder_{image_key}"]
            if "pretrained_encoder" in new_encoder_params:
                new_encoder_params = new_encoder_params["pretrained_encoder"]
            for k in new_encoder_params:
                if k in encoder_params:
                    new_encoder_params[k] = encoder_params[k]
                    print(f"replaced {module_key}.{image_key}.{k} from {label}")

    return agent.replace(state=agent.state.replace(params=new_params))


def load_resnet10_params(agent, image_keys=("image",), ckpt_path=None, public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if ckpt_path is not None:
        file_path = os.path.expandvars(os.path.expanduser(str(ckpt_path)))
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "ResNet-10 checkpoint not found at "
                f"{file_path}. Set train.resnet10_ckpt_path or RESNET10_CKPT_PATH."
            )
        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)
    elif not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = (
                f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            )
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        # This may fail when launching multiple jobs at the same time, so we try to load the file again
        # TODO: find a better way to do this
        success_load = False
        while not success_load:
            try:
                with open(file_path, "rb") as f:
                    encoder_params = pkl.load(f)
                success_load = True
            except:
                import time

                time.sleep(1)
                print("Failed to load ResNet-10 weights, retrying...")

    param_count = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(f"Loaded {param_count / 1e6}M parameters from ResNet-10 pretrained on ImageNet-1K")

    return _replace_resnet_encoder_params(agent, image_keys, encoder_params, "ResNet-10")


def load_resnet18_params(agent, image_keys=("image",), ckpt_path=None):
    """Load converted ImageNet ResNet-18 conv/GN-affine params into PLD encoders.

    The source checkpoint is torchvision ResNet-18 converted to Flax by
    jax-resnet. PLD's ResNet uses GroupNorm instead of BatchNorm, so running
    batch statistics are intentionally ignored; compatible conv kernels and
    scale/bias affine parameters are copied.
    """
    if ckpt_path is None:
        ckpt_path = os.environ.get("RESNET18_CKPT_PATH")
    if not ckpt_path:
        raise FileNotFoundError(
            "Set train.resnet18_ckpt_path or the RESNET18_CKPT_PATH environment variable."
        )
    ckpt_path = os.path.expandvars(os.path.expanduser(str(ckpt_path)))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "ResNet-18 checkpoint not found at "
            f"{ckpt_path}. Set train.resnet18_ckpt_path or RESNET18_CKPT_PATH."
        )

    with open(ckpt_path, "rb") as f:
        payload = pkl.load(f)
    variables = payload["variables"] if isinstance(payload, dict) and "variables" in payload else payload
    source_params = variables["params"]

    def copy_bn(dst_norm, src_bn):
        dst_norm["scale"] = src_bn["scale"]
        dst_norm["bias"] = src_bn["bias"]

    encoder_params = {}
    stem = source_params["layers_0"]["ConvBlock_0"]
    encoder_params["conv_init"] = {"kernel": stem["Conv_0"]["kernel"]}
    encoder_params["norm_init"] = {}
    copy_bn(encoder_params["norm_init"], stem["BatchNorm_0"])

    for block_idx in range(8):
        src = source_params[f"layers_{block_idx + 2}"]
        dst = {
            "Conv_0": {"kernel": src["ConvBlock_0"]["Conv_0"]["kernel"]},
            "MyGroupNorm_0": {},
            "Conv_1": {"kernel": src["ConvBlock_1"]["Conv_0"]["kernel"]},
            "MyGroupNorm_1": {},
        }
        copy_bn(dst["MyGroupNorm_0"], src["ConvBlock_0"]["BatchNorm_0"])
        copy_bn(dst["MyGroupNorm_1"], src["ConvBlock_1"]["BatchNorm_0"])
        if "ResNetSkipConnection_0" in src:
            skip = src["ResNetSkipConnection_0"]["ConvBlock_0"]
            dst["conv_proj"] = {"kernel": skip["Conv_0"]["kernel"]}
            dst["norm_proj"] = {}
            copy_bn(dst["norm_proj"], skip["BatchNorm_0"])
        encoder_params[f"ResNetBlock_{block_idx}"] = dst

    param_count = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(
        f"Loaded {param_count / 1e6}M compatible parameters from ResNet-18 "
        f"checkpoint at {ckpt_path}"
    )
    return _replace_resnet_encoder_params(agent, image_keys, encoder_params, "ResNet-18")


def load_dinov3_params(agent, image_keys=("image",), ckpt_path=None):
    """Load pretrained DINOv3 weights into agent's encoder backbone."""
    import torch

    from serl_launcher.vision.dinov3_weight_converter import convert_torch_to_flax

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    new_params = agent.state.params
    for image_key in image_keys:
        backbone_params = new_params["modules_actor"]["encoder"][f"encoder_{image_key}"][
            "backbone"
        ]
        converted = convert_torch_to_flax(ckpt, backbone_params)
        # Update both actor and critic encoders (they share structure)
        for module_key in ("modules_actor", "modules_critic"):
            new_params[module_key]["encoder"][f"encoder_{image_key}"]["backbone"] = converted

    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent

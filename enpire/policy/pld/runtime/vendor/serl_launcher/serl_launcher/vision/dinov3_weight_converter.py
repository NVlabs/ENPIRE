# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert PyTorch DINOv3 state_dict to Flax params pytree."""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.vision.dinov3_flax import (
    DinoVisionTransformer,
)

# Model configs keyed by backbone size name.
MODEL_CONFIGS = {
    "small": dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        ffn_ratio=4.0,
    ),
    "base": dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        ffn_ratio=4.0,
    ),
    "large": dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        ffn_ratio=4.0,
    ),
    "giant": dict(
        embed_dim=1536,
        depth=40,
        num_heads=24,
        patch_size=16,
        ffn_ratio=4.0,
    ),
}

# Keys that exist in PyTorch checkpoints but are not needed for inference.
_SKIP_PREFIXES = ("head.",)
_SKIP_KEYS = {"mask_token"}
_SKIP_SUBSTRINGS = ("bias_mask",)


def _t(x: np.ndarray) -> jnp.ndarray:
    """Transpose 2-D weight (PyTorch [out, in] -> Flax [in, out])."""
    return jnp.array(x.T)


def _a(x: np.ndarray) -> jnp.ndarray:
    """Direct copy as jnp array."""
    return jnp.array(x)


def _should_skip(key: str) -> bool:
    if key in _SKIP_KEYS:
        return True
    if any(key.startswith(p) for p in _SKIP_PREFIXES):
        return True
    return any(s in key for s in _SKIP_SUBSTRINGS)


def convert_torch_to_flax(
    torch_state_dict: dict,
    flax_params: dict,
) -> dict:
    """Convert a PyTorch DINOv3 state_dict to a Flax params dict.

    Args:
        torch_state_dict: PyTorch state_dict (str keys -> numpy arrays
            or torch tensors).  Tensors are converted to numpy internally.
        flax_params: Initialized Flax params pytree used as a template
            for shape verification.

    Returns:
        Nested dict suitable for ``model.apply({'params': params}, ...)``.

    Raises:
        ValueError: If a converted weight shape does not match the
            corresponding Flax template shape.
    """

    def _np(key: str) -> np.ndarray:
        v = torch_state_dict[key]
        if hasattr(v, "numpy"):
            return v.detach().cpu().float().numpy()
        return np.asarray(v, dtype=np.float32)

    # Auto-detect depth from the state_dict block keys.
    block_indices = set()
    for k in torch_state_dict:
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            block_indices.add(idx)
    depth = max(block_indices) + 1 if block_indices else 0

    # Auto-detect layerscale from the state_dict.
    has_layerscale = any(
        k.startswith("blocks.") and ".ls1.gamma" in k for k in torch_state_dict
    )

    # Auto-detect storage tokens.
    has_storage = "storage_tokens" in torch_state_dict

    # Auto-detect masked K bias (LinearKMaskedBias).
    has_mask_k_bias = any("bias_mask" in k for k in torch_state_dict)

    params: dict = {}
    converted_count = 0

    # --- patch_embed ---
    # PyTorch Conv2d weight: [C_out, C_in, kH, kW]
    # Flax Conv kernel:      [kH, kW, C_in, C_out]
    conv_w = _np("patch_embed.proj.weight")  # [D, 3, 16, 16]
    conv_w = conv_w.transpose(2, 3, 1, 0)  # [16, 16, 3, D]
    params["patch_embed"] = {
        "proj": {
            "kernel": jnp.array(conv_w),
            "bias": _a(_np("patch_embed.proj.bias")),
        }
    }
    converted_count += 2

    # --- cls_token ---
    params["cls_token"] = _a(_np("cls_token"))  # [1, 1, D]
    converted_count += 1

    # --- storage_tokens ---
    if has_storage:
        params["storage_tokens"] = _a(_np("storage_tokens"))  # [1, S, D]
        converted_count += 1

    # --- rope_embed ---
    params["rope_embed"] = {
        "periods": _a(_np("rope_embed.periods")),
    }
    converted_count += 1

    # --- transformer blocks ---
    for i in range(depth):
        prefix = f"blocks.{i}"
        blk: dict = {}

        # norm1
        blk["norm1"] = {
            "scale": _a(_np(f"{prefix}.norm1.weight")),
            "bias": _a(_np(f"{prefix}.norm1.bias")),
        }
        converted_count += 2

        # attention
        qkv_bias = _np(f"{prefix}.attn.qkv.bias")
        if has_mask_k_bias:
            # DINOv3 LinearKMaskedBias zeros out K bias at runtime.
            # Apply the same mask here so Flax Dense uses correct bias.
            D = qkv_bias.shape[0] // 3
            qkv_bias[D : 2 * D] = 0.0
        blk["attn"] = {
            "qkv": {
                "kernel": _t(_np(f"{prefix}.attn.qkv.weight")),
                "bias": _a(qkv_bias),
            },
            "proj": {
                "kernel": _t(_np(f"{prefix}.attn.proj.weight")),
                "bias": _a(_np(f"{prefix}.attn.proj.bias")),
            },
        }
        converted_count += 4

        # layer scale 1
        if has_layerscale:
            blk["ls1"] = {
                "gamma": _a(_np(f"{prefix}.ls1.gamma")),
            }
            converted_count += 1

        # norm2
        blk["norm2"] = {
            "scale": _a(_np(f"{prefix}.norm2.weight")),
            "bias": _a(_np(f"{prefix}.norm2.bias")),
        }
        converted_count += 2

        # mlp
        blk["mlp"] = {
            "fc1": {
                "kernel": _t(_np(f"{prefix}.mlp.fc1.weight")),
                "bias": _a(_np(f"{prefix}.mlp.fc1.bias")),
            },
            "fc2": {
                "kernel": _t(_np(f"{prefix}.mlp.fc2.weight")),
                "bias": _a(_np(f"{prefix}.mlp.fc2.bias")),
            },
        }
        converted_count += 4

        # layer scale 2
        if has_layerscale:
            blk["ls2"] = {
                "gamma": _a(_np(f"{prefix}.ls2.gamma")),
            }
            converted_count += 1

        params[f"blocks_{i}"] = blk

    # --- final norm ---
    params["norm"] = {
        "scale": _a(_np("norm.weight")),
        "bias": _a(_np("norm.bias")),
    }
    converted_count += 2

    # --- shape verification against Flax template ---
    _verify_shapes(params, flax_params)

    # --- summary ---
    original_count = sum(1 for k in torch_state_dict if not _should_skip(k))
    skipped = [k for k in torch_state_dict if _should_skip(k)]
    print(
        f"Converted {converted_count} params from PyTorch "
        f"({original_count} non-skip keys in checkpoint)."
    )
    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped}")

    return params


def _verify_shapes(converted: dict, template: dict, path: str = "") -> None:
    """Recursively verify converted param shapes match the template."""
    for key in template:
        full_path = f"{path}/{key}" if path else key
        if key not in converted:
            raise ValueError(f"Missing key in converted params: {full_path}")
        t_val = template[key]
        c_val = converted[key]
        if isinstance(t_val, dict):
            if not isinstance(c_val, dict):
                raise ValueError(f"Expected dict at {full_path}, got {type(c_val)}")
            _verify_shapes(c_val, t_val, full_path)
        else:
            t_shape = jnp.asarray(t_val).shape
            c_shape = jnp.asarray(c_val).shape
            if t_shape != c_shape:
                raise ValueError(
                    f"Shape mismatch at {full_path}: converted {c_shape} vs template {t_shape}"
                )


def load_dinov3_flax(
    backbone_size: str = "small",
    ckpt_path: str | None = None,
    n_storage_tokens: int = 0,
    layerscale_init: float | None = None,
    norm_eps: float = 1e-6,
    rope_normalize_coords: str = "separate",
    mask_k_bias: bool = False,
) -> Tuple[DinoVisionTransformer, dict]:
    """Load a DINOv3 model with converted weights.

    Args:
        backbone_size: one of "small", "base", "large", "giant".
        ckpt_path: path to PyTorch checkpoint (``.pth``).  If None the
            model is returned with random init params (useful for testing).
        n_storage_tokens: number of storage/register tokens.
        layerscale_init: LayerScale init value, or None to disable.
        norm_eps: LayerNorm epsilon.
        rope_normalize_coords: RoPE coordinate normalization mode.

    Returns:
        (model, params) where ``model`` is a
        :class:`DinoVisionTransformer` and ``params`` is the Flax params
        dict ready for ``model.apply({'params': params}, x)``.
    """
    cfg = MODEL_CONFIGS[backbone_size].copy()
    cfg["n_storage_tokens"] = n_storage_tokens
    cfg["layerscale_init"] = layerscale_init
    cfg["norm_eps"] = norm_eps
    cfg["rope_normalize_coords"] = rope_normalize_coords
    cfg["mask_k_bias"] = mask_k_bias

    model = DinoVisionTransformer(**cfg)

    # Always initialize to get the template param tree.
    template_params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 224, 224, 3)),
    )["params"]

    if ckpt_path is not None:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Some checkpoints wrap state_dict in a "model" key.
        if "model" in ckpt:
            ckpt = ckpt["model"]
        params = convert_torch_to_flax(ckpt, template_params)
    else:
        params = template_params

    return model, params


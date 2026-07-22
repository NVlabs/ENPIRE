# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Vision Transformer in Flax/JAX.

Numerically equivalent port of the PyTorch DINOv3 ViT.
All tensor layouts follow Flax conventions (NHWC for images).
"""

import math
from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------
# RoPE helpers
# --------------------------------------------------------------------------


def rope_rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """[-x2, x1] rotation used by RoPE."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_apply(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    # Match PyTorch's bf16 RoPE arithmetic under jit.  Without explicit
    # round/barrier points XLA fuses the multiply-add and rounds differently,
    # which can flip sharp attention probabilities after only one block.
    out_dtype = x.dtype
    x_cos = (x * cos).astype(out_dtype)
    x_sin = (rope_rotate_half(x) * sin).astype(out_dtype)
    x_cos = jax.lax.optimization_barrier(x_cos)
    x_sin = jax.lax.optimization_barrier(x_sin)
    return (x_cos + x_sin).astype(out_dtype)


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, H, W, 3) -> (B, num_patches, D)."""

    embed_dim: int = 768
    patch_size: int = 16
    in_chans: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        """Returns (tokens [B, H*W, D], (H_patches, W_patches))."""
        ps = self.patch_size
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(ps, ps),
            strides=(ps, ps),
            padding="VALID",
            use_bias=True,
            precision=jax.lax.Precision.HIGHEST,
            name="proj",
        )(x)  # [B, Hp, Wp, D]
        B, Hp, Wp, D = x.shape
        x = x.reshape(B, Hp * Wp, D)
        return x, (Hp, Wp)


class RoPE(nn.Module):
    """2D axial Rotary Position Embedding (matches PyTorch DINOv3).

    Stores ``periods`` as a model parameter so that they are loaded from
    the checkpoint.  Forward produces ``(sin, cos)`` each of shape
    ``[H*W, D_head]`` computed in the same bf16 path as the official model.
    """

    d_head: int
    normalize_coords: str = "separate"
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, H: int, W: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # DINOv3 RoPE periods are a deterministic checkpoint buffer.  Computing
        # the bf16 sin/cos inside XLA changes the values under jit, so compute
        # the same deterministic bf16 table at trace time and embed it as a
        # constant for compiled actor/critic execution.
        self.param(
            "periods",
            lambda rng, shape: jnp.ones(shape, dtype=jnp.float32),
            (self.d_head // 4,),
        )
        with jax.ensure_compile_time_eval():
            periods = jnp.asarray(100.0, dtype=self.dtype) ** (
                2
                * jnp.arange(self.d_head // 4, dtype=self.dtype)
                / (self.d_head // 2)
            )

            if self.normalize_coords == "separate":
                coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / H
                coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / W
            elif self.normalize_coords == "max":
                m = max(H, W)
                coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / m
                coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / m
            elif self.normalize_coords == "min":
                m = min(H, W)
                coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / m
                coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / m
            else:
                raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

            # meshgrid -> [H, W, 2] -> [HW, 2], range [0,1] -> [-1,1]
            grid_h, grid_w = jnp.meshgrid(coords_h, coords_w, indexing="ij")
            coords = jnp.stack([grid_h, grid_w], axis=-1)  # [H, W, 2]
            coords = coords.reshape(H * W, 2)
            coords = 2.0 * coords - 1.0

            # angles: [HW, 2, D_head//4]
            angles = 2.0 * math.pi * coords[:, :, None] / periods[None, None, :]
            angles = angles.reshape(H * W, self.d_head // 2)  # [HW, D//2]
            angles = jnp.tile(angles, (1, 2))  # [HW, D_head]

            cos = jnp.cos(angles)
            sin = jnp.sin(angles)
        return sin, cos  # [HW, D_head]


class LayerScale(nn.Module):
    """Learnable per-channel scale (gamma)."""

    dim: int
    init_value: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.param(
            "gamma",
            lambda rng, shape: jnp.full(shape, self.init_value),
            (self.dim,),
        )
        return x * gamma


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE support."""

    embed_dim: int
    num_heads: int
    qkv_bias: bool = True
    proj_bias: bool = True
    mask_k_bias: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope=None,
    ) -> jnp.ndarray:
        B, N, C = x.shape
        head_dim = self.embed_dim // self.num_heads

        qkv = nn.Dense(
            self.embed_dim * 3,
            use_bias=self.qkv_bias,
            precision=jax.lax.Precision.HIGHEST,
            name="qkv",
        )(x)  # [B, N, 3*C]

        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        q, k, v = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )  # each [B, N, heads, head_dim]
        q = q.transpose(0, 2, 1, 3)  # [B, heads, N, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if rope is not None:
            sin, cos = rope
            rope_dtype = sin.dtype
            q_dtype, k_dtype = q.dtype, k.dtype
            q = q.astype(rope_dtype)
            k = k.astype(rope_dtype)

            prefix = N - sin.shape[0]  # CLS + storage tokens
            # Only apply RoPE to patch tokens
            q_prefix = q[:, :, :prefix, :]
            q_patch = rope_apply(q[:, :, prefix:, :], sin[None, None], cos[None, None])
            q = jnp.concatenate([q_prefix, q_patch], axis=2)

            k_prefix = k[:, :, :prefix, :]
            k_patch = rope_apply(k[:, :, prefix:, :], sin[None, None], cos[None, None])
            k = jnp.concatenate([k_prefix, k_patch], axis=2)

            q = q.astype(q_dtype)
            k = k.astype(k_dtype)

        scale = head_dim**-0.5
        attn = (
            jnp.einsum(
                "bhnd,bhmd->bhnm",
                q,
                k,
                precision=jax.lax.Precision.HIGHEST,
            )
            * scale
        )
        attn = jax.nn.softmax(attn, axis=-1)
        x = jnp.einsum(
            "bhnm,bhmd->bhnd",
            attn,
            v,
            precision=jax.lax.Precision.HIGHEST,
        )

        x = x.transpose(0, 2, 1, 3)  # [B, N, heads, head_dim]
        x = x.reshape(B, N, self.embed_dim)

        x = nn.Dense(
            self.embed_dim,
            use_bias=self.proj_bias,
            precision=jax.lax.Precision.HIGHEST,
            name="proj",
        )(x)
        return x


class Mlp(nn.Module):
    """MLP: Dense -> GELU -> Dense."""

    in_features: int
    hidden_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_features,
            use_bias=True,
            precision=jax.lax.Precision.HIGHEST,
            name="fc1",
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(
            self.in_features,
            use_bias=True,
            precision=jax.lax.Precision.HIGHEST,
            name="fc2",
        )(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Transformer block: norm -> attn -> ls -> residual, repeat for MLP."""

    embed_dim: int
    num_heads: int
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    init_values: float | None = None
    norm_eps: float = 1e-6
    mask_k_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, rope=None) -> jnp.ndarray:
        # --- self-attention path ---
        residual = x
        x_norm = nn.LayerNorm(epsilon=self.norm_eps, name="norm1")(x)
        x_attn = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            mask_k_bias=self.mask_k_bias,
            name="attn",
        )(x_norm, rope=rope)
        if self.init_values is not None:
            x_attn = LayerScale(
                dim=self.embed_dim,
                init_value=self.init_values,
                name="ls1",
            )(x_attn)
        x = residual + x_attn

        # --- MLP path ---
        residual = x
        x_norm = nn.LayerNorm(epsilon=self.norm_eps, name="norm2")(x)
        mlp_hidden = int(self.embed_dim * self.ffn_ratio)
        x_mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=mlp_hidden,
            name="mlp",
        )(x_norm)
        if self.init_values is not None:
            x_mlp = LayerScale(
                dim=self.embed_dim,
                init_value=self.init_values,
                name="ls2",
            )(x_mlp)
        x = residual + x_mlp
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv3 Vision Transformer in Flax.

    Input:  [B, H, W, 3]  (NHWC)
    Output: dict with x_norm_clstoken, x_norm_patchtokens, etc.
    """

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    n_storage_tokens: int = 0
    layerscale_init: float | None = None
    norm_eps: float = 1e-6
    rope_normalize_coords: str = "separate"
    rope_dtype: jnp.dtype = jnp.bfloat16
    mask_k_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False):
        """Full forward pass. Returns normed feature dict."""
        x, _ = self._run_backbone(x)
        x_norm = nn.LayerNorm(epsilon=self.norm_eps, name="norm")(x)
        n_prefix = 1 + self.n_storage_tokens
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_storage_tokens": x_norm[:, 1:n_prefix],
            "x_norm_patchtokens": x_norm[:, n_prefix:],
            "x_prenorm": x,
        }

    @nn.compact
    def _run_backbone(self, x: jnp.ndarray):
        """Shared backbone: patch_embed → tokens → blocks.

        Returns (x, (Hp, Wp)) after all transformer blocks.
        """
        d_head = self.embed_dim // self.num_heads

        tokens, (Hp, Wp) = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            name="patch_embed",
        )(x)
        B = tokens.shape[0]

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        parts = [jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))]

        if self.n_storage_tokens > 0:
            storage_tokens = self.param(
                "storage_tokens",
                nn.initializers.normal(stddev=0.02),
                (1, self.n_storage_tokens, self.embed_dim),
            )
            parts.append(
                jnp.broadcast_to(
                    storage_tokens,
                    (B, self.n_storage_tokens, self.embed_dim),
                )
            )

        parts.append(tokens)
        x = jnp.concatenate(parts, axis=1)  # [B, 1+S+HW, D]

        rope_sincos = RoPE(
            d_head=d_head,
            normalize_coords=self.rope_normalize_coords,
            dtype=self.rope_dtype,
            name="rope_embed",
        )(Hp, Wp)

        for i in range(self.depth):
            x = SelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=self.ffn_ratio,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                init_values=self.layerscale_init,
                norm_eps=self.norm_eps,
                mask_k_bias=self.mask_k_bias,
                name=f"blocks_{i}",
            )(x, rope=rope_sincos)

        return x, (Hp, Wp)

    @nn.compact
    def get_intermediate_layers(
        self,
        x: jnp.ndarray,
        *,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        """Extract intermediate layer features (inference only).

        Args:
            x: [B, H, W, 3] input image.
            n: number of last layers (int) or explicit layer indices.
            reshape: if True return spatial [B, Hp, Wp, D] (NHWC).
            return_class_token: also return CLS token per layer.
            norm: apply final LayerNorm.

        Returns:
            Tuple of tensors (or tuple of (features, cls) pairs).
        """
        d_head = self.embed_dim // self.num_heads

        tokens, (Hp, Wp) = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            name="patch_embed",
        )(x)
        B = tokens.shape[0]

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        parts = [jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))]

        if self.n_storage_tokens > 0:
            storage_tokens = self.param(
                "storage_tokens",
                nn.initializers.normal(stddev=0.02),
                (1, self.n_storage_tokens, self.embed_dim),
            )
            parts.append(
                jnp.broadcast_to(
                    storage_tokens,
                    (B, self.n_storage_tokens, self.embed_dim),
                )
            )

        parts.append(tokens)
        h = jnp.concatenate(parts, axis=1)

        rope_sincos = RoPE(
            d_head=d_head,
            normalize_coords=self.rope_normalize_coords,
            dtype=self.rope_dtype,
            name="rope_embed",
        )(Hp, Wp)

        if isinstance(n, int):
            blocks_to_take = set(range(self.depth - n, self.depth))
        else:
            blocks_to_take = set(n)

        outputs = []
        for i in range(self.depth):
            h = SelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=self.ffn_ratio,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                init_values=self.layerscale_init,
                norm_eps=self.norm_eps,
                mask_k_bias=self.mask_k_bias,
                name=f"blocks_{i}",
            )(h, rope=rope_sincos)
            if i in blocks_to_take:
                outputs.append(h)

        if norm:
            norm_layer = nn.LayerNorm(epsilon=self.norm_eps, name="norm")
            outputs = [norm_layer(o) for o in outputs]

        n_prefix = 1 + self.n_storage_tokens
        class_tokens = [o[:, 0] for o in outputs]
        outputs = [o[:, n_prefix:] for o in outputs]

        if reshape:
            outputs = [o.reshape(B, Hp, Wp, -1) for o in outputs]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


# ------------------------------------------------------------------
# Pretrained encoder wrapper (for SAC pipeline integration)
# ------------------------------------------------------------------


class PreTrainedDINOv3Encoder(nn.Module):
    """Frozen DINOv3 backbone + trainable bottleneck for SAC pixel encoder.

    Follows the same ``__call__(observations, encode, train)`` interface as
    ``PreTrainedResNetEncoder`` so ``EncodingWrapper`` can call it.
    """

    backbone: DinoVisionTransformer
    pooling_method: str = "cls"  # "cls" | "avg_patches"
    bottleneck_dim: int | None = 256
    freeze_backbone: bool = True

    @nn.compact
    def __call__(self, observations, encode=True, train=True):
        x = observations  # [B, H, W, C] or [H, W, C] (no batch during init)
        if encode:
            # Add batch dim if missing (EncodingWrapper stacking can drop it)
            no_batch = x.ndim == 3
            if no_batch:
                x = x[None]

            in_channels = x.shape[-1]
            if in_channels > 3:
                # Per-frame encoding for stacked frames
                n_frames = in_channels // 3
                B = x.shape[0]
                frames = x.reshape(B * n_frames, *x.shape[1:3], 3)
                out = self.backbone(frames, train=False)
                cls = out["x_norm_clstoken"]  # [B*T, D]
                x = cls.reshape(B, n_frames * cls.shape[-1])
            else:
                out = self.backbone(x, train=False)
                if self.pooling_method == "cls":
                    x = out["x_norm_clstoken"]  # [B, D]
                else:
                    x = out["x_norm_patchtokens"].mean(axis=1)
            if self.freeze_backbone:
                x = jax.lax.stop_gradient(x)  # freeze backbone

            if no_batch:
                x = x[0]

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        return x


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------


def dinov3_vits16(**kwargs) -> DinoVisionTransformer:
    defaults = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        ffn_ratio=4.0,
    )
    defaults.update(kwargs)
    return DinoVisionTransformer(**defaults)


def dinov3_vitb16(**kwargs) -> DinoVisionTransformer:
    defaults = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        ffn_ratio=4.0,
    )
    defaults.update(kwargs)
    return DinoVisionTransformer(**defaults)


def dinov3_vitl16(**kwargs) -> DinoVisionTransformer:
    defaults = dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        ffn_ratio=4.0,
    )
    defaults.update(kwargs)
    return DinoVisionTransformer(**defaults)


def dinov3_vitg16(**kwargs) -> DinoVisionTransformer:
    defaults = dict(
        embed_dim=1536,
        depth=40,
        num_heads=24,
        patch_size=16,
        ffn_ratio=4.0,
    )
    defaults.update(kwargs)
    return DinoVisionTransformer(**defaults)


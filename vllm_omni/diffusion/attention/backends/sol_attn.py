# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.layer import Attention

SOL_MORTON_ORDER = "sol_morton_order"


@lru_cache(maxsize=8)
def get_morton_indices(grid: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raster video tokens to 3D Z-order (W, H, T bits) and back."""
    frames, height, width = grid
    raster = torch.arange(frames * height * width, device="cpu")
    w, h, t = raster % width, raster // width % height, raster // (height * width)
    code = torch.zeros_like(raster)
    for bit in range((max(grid) - 1).bit_length()):
        for axis, coord in enumerate((w, h, t)):
            code |= ((coord >> bit) & 1) << (3 * bit + axis)
    order = code.argsort()
    return order.to(device), order.argsort().to(device)


def wan_sol_layout(last_attention: Attention) -> dict[str, bool]:
    """All Wan blocks share a profile. With a dense prefix, the last layer is sparse if any layer is."""
    impl, _, _ = last_attention.effective_attention()
    if not isinstance(impl, SolAttnImpl):
        return {}
    step = get_forward_context().denoise_step_idx
    return {SOL_MORTON_ORDER: not impl.dense_layer and step is not None and step >= impl.dense_steps}


class SolAttnBackend(AttentionBackend):
    supported_platforms = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        # Load the provider before CANN snapshots its custom-op registry.
        try:
            from mindiesd import sol_attn
        except ImportError as exc:
            raise ValueError(
                "SOL_ATTN requires a MindIE-SD build with mindiesd.sol_attn (Ascend/MindIE-SD#929)."
            ) from exc

        if not callable(sol_attn):
            raise ValueError("MindIE-SD must export a callable mindiesd.sol_attn.")

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl[AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        qkv_layout: str | None = None,
        role: str = "self",
        prefix: str = "",
        backend_kwargs: dict[str, float | int] | None = None,
        **kwargs: object,
    ) -> None:
        if (
            role != "self"
            or causal
            or qkv_layout != "BSND"
            or head_size != 128
            or num_kv_heads not in (None, num_heads)
        ):
            raise ValueError("SOL_ATTN requires noncausal Wan self-attention with BSND layout and head size 128.")
        config = get_current_diffusion_config_or_none()
        if config is not None:
            if config.dtype != torch.bfloat16:
                raise ValueError("SOL_ATTN requires BF16. Set the diffusion dtype to bfloat16.")
            if config.parallel_config.world_size != 1:
                raise ValueError("SOL_ATTN currently supports single-device Wan inference.")
            if config.diffusion_use_graph or config.cache_backend not in (None, "none"):
                raise ValueError("SOL_ATTN step-dependent token ordering requires uncached, non-graphed DiT execution.")
        options = backend_kwargs or {}
        self.tau = options.get("tau", 1.0)
        self.scale = softmax_scale
        self.dense_steps = options.get("dense_steps", 4)
        self.dense_layer = extract_layer_index(prefix) < options.get("dense_layers", 1)
        from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

        self.dense = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads, head_size=head_size, softmax_scale=softmax_scale, qkv_layout=qkv_layout
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None or SOL_MORTON_ORDER not in attn_metadata.extra or attn_metadata.attn_mask is not None:
            raise ValueError("SOL_ATTN needs an unmasked Wan video token sequence with a declared token order.")
        if self.dense_layer or not attn_metadata.extra[SOL_MORTON_ORDER]:
            return self.dense.forward_npu(query, key, value, attn_metadata)
        from mindiesd import sol_attn

        return sol_attn(query.contiguous(), key.contiguous(), value.contiguous(), scale=self.scale, tau=self.tau)

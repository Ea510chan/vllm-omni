# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnBackend, SolAttnImpl, wan_sol_layout
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig, SolAttnSpec
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import _morton_block_inputs, _morton_indices

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sol_is_explicit_and_scoped_to_self(monkeypatch):
    assert AttentionConfig().resolve_with_source("self")[0] is None
    config = AttentionConfig(per_role={"self": {"backend": "SOL_ATTN", "sol_attn": {"tau": 1.25}}})
    assert config.resolve_with_source("cross")[0] is None
    spec, _ = config.resolve_with_source("self")
    assert spec.backend_kwargs() == {"tau": 1.25, "dense_layers": 1, "dense_steps": 4}
    assert DiffusionAttentionBackendEnum.SOL_ATTN.get_class() is SolAttnBackend
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    with pytest.raises(ValueError, match="Ascend/MindIE-SD#929"):
        SolAttnBackend.validate_available()


@pytest.mark.parametrize("options", [{"tau": float("nan")}, {"dense_steps": -1}, {"dense_layers": 1.5}])
def test_sol_options(options):
    with pytest.raises(ValueError):
        SolAttnSpec(**options)
    with pytest.raises(ValueError, match="require the SOL_ATTN"):
        AttentionSpec(backend="FLASH_ATTN", sol_attn={})


def test_sol_rejects_fp16_at_startup():
    with set_current_diffusion_config(OmniDiffusionConfig(dtype=torch.float16)):
        with pytest.raises(ValueError, match="requires BF16"):
            SolAttnImpl(2, 128, 0.2, prefix="blocks.0.attn1", qkv_layout="BSND")


@pytest.mark.parametrize(
    ("step", "layer", "sparse"), [(None, 1, False), (3, 29, False), (4, 0, False), (4, 1, True), (49, 29, True)]
)
def test_sol_step_and_layer_dispatch(monkeypatch, step, layer, sparse):
    spec = AttentionSpec(backend="SOL_ATTN")
    impl = SolAttnImpl(
        2, 128, 0.2, prefix=f"blocks.{layer}.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs()
    )
    provider = ModuleType("mindiesd")
    kernel = Mock(side_effect=lambda q, k, v, **kw: q + 1)
    monkeypatch.setattr(provider, "sol_attn", kernel, raising=False)
    monkeypatch.setitem(sys.modules, "mindiesd", provider)
    dense = Mock(side_effect=lambda q, *args: q)
    monkeypatch.setattr(impl.dense, "forward_npu", dense)
    q = torch.zeros(1, 2, 4, 128, dtype=torch.bfloat16).transpose(1, 2)
    last_attention = Mock(spec=Attention)
    last_impl = SolAttnImpl(
        2, 128, 0.2, prefix="blocks.29.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs()
    )
    last_attention.effective_attention.return_value = (last_impl, SolAttnBackend, spec)
    with set_forward_context(denoise_step_idx=step):
        meta = AttentionMetadata(extra=wan_sol_layout(last_attention))
    output = impl.forward_npu(q, q, q, meta)
    assert dense.call_count == int(not sparse)
    assert kernel.call_count == int(sparse)
    torch.testing.assert_close(output, q + int(sparse))
    if sparse:
        assert all(arg.is_contiguous() for arg in kernel.call_args.args)
        assert kernel.call_args.kwargs == {"scale": 0.2, "tau": 1.0}
    with pytest.raises(ValueError, match="declared token order"):
        impl.forward_npu(q, q, q, AttentionMetadata())


def test_wan_sol_layout_follows_active_backend_and_all_dense_layers():
    last_attention = Mock(spec=Attention)
    spec = AttentionSpec(backend="SOL_ATTN", sol_attn={"dense_layers": 30, "dense_steps": 0})
    impl = SolAttnImpl(2, 128, 0.2, prefix="blocks.29.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs())
    last_attention.effective_attention.side_effect = [(impl, SolAttnBackend, spec), (impl.dense, None, None)]
    with set_forward_context(denoise_step_idx=10):
        assert wan_sol_layout(last_attention) == {"sol_morton_order": False}
        assert wan_sol_layout(last_attention) == {}


def test_morton_alignment_and_dense_round_trip():
    grid = (2, 2, 3)
    order, inverse = _morton_indices(grid, torch.device("cpu"))
    assert order.tolist() == [0, 1, 3, 4, 6, 7, 9, 10, 2, 5, 8, 11]
    assert torch.equal(order[inverse], torch.arange(12))
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(2, 12, 8, generator=generator)
    cos, sin = torch.randn(2, 1, 12, 1, 8, generator=generator)
    per_token = torch.randn(2, 12, 6, 8, generator=generator)
    hidden, (mc, ms), mt = _morton_block_inputs(tokens, (cos, sin), per_token, order)
    torch.testing.assert_close(mc, cos[:, order])
    torch.testing.assert_close(ms, sin[:, order])
    torch.testing.assert_close(mt, per_token[:, order])
    reference = scaled_dot_product_attention(
        tokens + cos.squeeze(2), tokens + sin.squeeze(2), tokens + per_token[:, :, 0]
    )
    actual = scaled_dot_product_attention(hidden + mc.squeeze(2), hidden + ms.squeeze(2), hidden + mt[:, :, 0])
    torch.testing.assert_close(actual[:, inverse], reference)

    per_sample = per_token[:, 0]
    assert _morton_block_inputs(tokens, (cos, sin), per_sample, order)[2] is per_sample

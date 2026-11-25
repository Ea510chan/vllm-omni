"""External observer for one benchmark; disabled unless OMNI_SOL_BENCH is set."""
import importlib.abc
import importlib.machinery
import json
import os
import sys
from unittest.mock import patch

target = "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2"

def instrument(module):
    import torch
    from contextlib import ExitStack
    from functools import wraps
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention, WanTransformer3DModel

    diffuse = module.Wan22Pipeline.diffuse
    @wraps(diffuse)
    def run(self, *args, **kwargs):
        kind = os.environ["OMNI_SOL_BENCH"]
        records = []
        counts = dict(self_calls=0, sol_calls=0, exact=0, sol_possible=0, total_possible=0)
        with ExitStack() as stack:
            if kind == "stats":
                # Pinned 910B m128 ABI (MindIE-SD 69852419). Read the actual route
                # counts consumed by the kernel, excluding its padded query tiles.
                op = torch.ops.mindiesd.sol_attn_fwd_m128
                attn = WanSelfAttention.forward
                def self_forward(layer, hidden, *args, **kwargs):
                    batch, seq, _ = hidden.shape
                    counts["self_calls"] += 1
                    counts["total_possible"] += batch * layer.num_heads * ((seq + 127)//128) * ((seq + 63)//64)
                    return attn(layer, hidden, *args, **kwargs)
                def m128(*buffers):
                    q = buffers[0]
                    batch, seq, heads, _ = q.shape
                    tiles = (seq + 127)//128
                    nkeys = (seq + 63)//64
                    exact_count = buffers[5][..., :tiles].cpu()
                    assert int(exact_count.min()) >= 0 and int(exact_count.max()) <= nkeys
                    counts["sol_calls"] += 1
                    counts["sol_possible"] += batch * heads * tiles * nkeys
                    counts["exact"] += int(exact_count.sum())
                    return op(*buffers)
                stack.enter_context(patch.object(WanSelfAttention, "forward", self_forward))
                stack.enter_context(patch.object(torch.ops.mindiesd, "sol_attn_fwd_m128", m128))
            else:
                forward = WanTransformer3DModel.forward
                @wraps(forward)
                def timed(model, *inputs, **kwargs):
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin.record()
                    result = forward(model, *inputs, **kwargs)
                    end.record()
                    records.append((begin, end))
                    return result
                stack.enter_context(patch.object(WanTransformer3DModel, "forward", timed))
            result = diffuse(self, *args, **kwargs)
        if records:
            records[-1][1].synchronize()
            counts = {"dit_forwards": len(records), "dit_npu_seconds": sum(a.elapsed_time(b) for a, b in records)/1000}
        print("SOL_BENCH " + json.dumps(counts), flush=True)
        return result
    module.Wan22Pipeline.diffuse = run

class Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def create_module(self, spec):
        return self.wrapped.create_module(spec)
    def exec_module(self, module):
        self.wrapped.exec_module(module)
        instrument(module)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target_module=None):
        if fullname == target:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec

if os.environ.get("OMNI_SOL_BENCH") in ("timing", "stats"):
    sys.meta_path.insert(0, Finder())

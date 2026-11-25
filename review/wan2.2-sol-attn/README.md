# Wan2.2 TI2V-5B Sol-Attn review evidence

[Feature commit](https://github.com/Ea510chan/vllm-omni/commit/b1bba59fbae9e6d6c16c99a3ca751160da5c3ef8) · [Reproduction commands](../../docs/user_guide/diffusion/attention_backends/sol_attn.md) · [Independent agent review](agent-review.md)

The single-card comparison used a shared Ascend 910B3 with BF16, 1280 × 704, 121 frames, 50 UniPC steps, CFG 5 and seed 1024. Sol used diagonal thresholding at tau 1.0. The first four steps and first layer stayed dense. MindIE-SD [PR 929](https://gitcode.com/Ascend/MindIE-SD/pull/929) was tested at commit `6985241966212ee36cc1f10a222e03e8154330e9`.

| Metric | Dense | Sol diag |
| --- | ---: | ---: |
| DiT (s) | 418.52 | 247.05 |
| Generation request (s) | 463.07 | 288.07 |
| Launch through MP4 export (s) | 574.72 | 397.45 |
| PSNR vs. dense (dB, mean per frame) | Reference | 21.38 |
| SSIM vs. dense (mean per frame) | Reference | 0.8207 |

Timing used one measured request per mode after warm-up. A separate run measured sparsity of 68.98% within the 2,668 Sol calls and 61.34% across all 3,000 self-attention calls. Quality was computed on the uncompressed decoded frames. The linked video is a scaled, synchronized preview of the generated outputs, with dense on the left and diag on the right.

- [Generated comparison video](dense-vs-diag.mp4)
- [Settings, prompt, staged timings, versions, quality and routing counts](summary.json)
- [Dense timing](dense-timed.log), [diag timing](diag-timed.log) and [independent routing capture](diag-routing.log). These are measurement excerpts. The [external observer](route_and_timing_observer.py) documents the timing and block-count methods for the pinned provider revision.
- [Focused tests including the 910B3 end-to-end case](review-tests.log) and [pre-commit checks](review-precommit.log). In the broader suite [six CPU VAE cases failed](pytest-wan.log) and [reproduced on the base revision](baseline-six.log).

Codex assisted with implementation, tests, documentation and an independent agent review. Human review and DCO sign-off are pending.

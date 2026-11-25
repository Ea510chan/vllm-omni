# Wan2.2 TI2V-5B Sol-Attn review evidence

[50-step benchmark commit](https://github.com/Ea510chan/vllm-omni/commit/b1bba59fbae9e6d6c16c99a3ca751160da5c3ef8) · [Reproduction commands](../../docs/user_guide/diffusion/attention_backends/sol_attn.md) · [Independent agent review](agent-review.md)

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


## Routing refactor

[Refactored feature commit](https://github.com/Ea510chan/vllm-omni/commit/a001cc8544ef8c5051233ca5a746ee70afa35f75). The 50-step estimates above come from the earlier commit. This revision was validated with focused tests and a separate three-step routing probe on one Ascend 910B3.

- [168 tests passed](refactor-tests.log), including the real two-step Wan generation case, dense-prefix boundaries, attention configuration, Morton round trip and Wan scheduled-attention compile tests. All [applicable pre-commit hooks passed](refactor-precommit.log).
- A second 832 × 480, 17-frame generation used CFG 5, seed 1024 and a service profile selecting Sol only at zero-based step 1 (tau 1, dense_steps 0, dense_layers 1). Each step ran two DiT forwards. The [observed provider calls and layout requests](schedule-routes.log) were 0, 29 and 0 Sol calls per forward and 0, 1 and 0 Morton index requests. The output MP4 decoded to 17 frames.
- An independent Codex review checked the empty-stage guard, dense baseline, dense-only layers and active schedule selection. Its local type-annotation suggestion was applied. Human review and DCO sign-off are pending.

## Morton index cache

[Feature commit](https://github.com/Ea510chan/vllm-omni/commit/758442d34a9b235c307afba124e693eec65c67bd) moves the index generator and its eight-entry cache into the Sol backend. Wan applies the token order around the block stack and constructs independent metadata for each block. An independent Codex review confirmed the permutation algorithm and cache key are unchanged.

[168 tests passed](morton-move-tests.log), including a cache-reuse assertion and the 910B3 end-to-end case. [Pre-commit checks passed](morton-move-precommit.log). The three-step dense/diag/dense experiment reproduced the six calls in the [previous routing trace](schedule-routes.log). The 17 decoded RGB24 frames matched the previous output byte for byte (SHA-256 `c2011a6d9bf209c5e7d1339a889f1270a0581c34b60909ca9abcaf498d2fcc67`).

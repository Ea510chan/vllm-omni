# Wan2.2 TI2V-5B 50-step benchmark after moving the Morton cache

[Feature revision](https://github.com/Ea510chan/vllm-omni/commit/758442d34a9b235c307afba124e693eec65c67bd) · [Historical 50-step baseline](../summary.json)

Both modes ran sequentially on physical card 2 of the same Ascend 910B3 host. One request per mode was timed after engine warm-up. Each used the same prompt, BF16 weights, 1280 × 704 resolution, 121 frames, 24 fps, CFG 5, flow shift 5, seed 1024 and 50 UniPC steps. Sol used tau 1.0 with four initial dense steps and one initial dense layer. MindIE-SD PR 929 at `69852419` exposes the diag threshold.

| Stage | Dense (s) | Sol diag (s) |
| --- | ---: | ---: |
| Weight load | 19.681 | 22.870 |
| Text encoding | 0.137 | 0.138 |
| DiT (100 forwards) | 385.047 | 218.041 |
| Denoising loop | 385.467 | 218.426 |
| VAE decode | 30.224 | 30.070 |
| Generation request | 419.220 | 251.827 |
| MP4 export | 6.064 | 6.610 |
| Launch through MP4 export | 528.055 | 371.792 |

Dense/Sol diag speedup was 1.766× in DiT, 1.665× for the generation request, and 1.420× from launch through MP4 export. Launch includes initialization, weight load and engine warm-up. Stage intervals overlap. Shared-host single samples limit comparisons of small timing differences.

Relative to dense, mean per-frame PSNR was 21.3832 dB and SSIM was 0.820696 across the 121 pre-MP4 FP32 RGB frames. An independent route capture counted 2,668 Sol and 332 dense self-attention invocations. The approximate block fractions were 68.9759% within Sol and 61.3426% across all 3,000 calls.

The previous DiT measurements were 418.519 s for dense and 247.053 s for diag. Both modes matched the respective previous uncompressed FP32 output byte for byte. Cross-run hashes, quality and route-count comparisons are in the JSON report.

[Settings, metrics and output hashes](summary.json) · [Dense timing](dense-timed.log) · [Diag timing](diag-timed.log) · [Route statistics](diag-routing.log)

[Side-by-side preview of the verified matching outputs, dense on the left and Sol diag on the right](../dense-vs-diag.mp4)

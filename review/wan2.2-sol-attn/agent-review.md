# Independent agent review of Wan2.2 Sol-Attn

On 2026-11-25, the independent review covered tracked and newly added files on
`features/wan2.2-sol-attn` against base `a57d6d1a7`. The final review covered
10 files with content fingerprint `90cc1a697158937d9a8571f7bea718733aaa83dabfa9ecb781bb7a2856413efc`.

| Initial finding | Resolution |
| --- | --- |
| MindIE-SD Sol supports BF16. With FP16 selected, the previous implementation could finish the dense prefix and fail on the first Sol call. | Reject incompatible dtypes during initialization and cover FP16 with a regression test. |
| The two-step hardware example depended on an external script. | Add a single-A2 case to the existing offline test. The first step uses dense attention and the second enables Sol. Check the output shape, finite values and pixel variation. |
| The documented CLI defaults did not reproduce the 121-frame experiment. | Provide paired dense and diag commands with explicit resolution, frame count, seed, CFG, model input and operator revision. |

The follow-up review found the changes ready for human review and a draft PR.
The two-step test checks the generated output. Actual Sol call counts come from
a separate 50-step routing run with fixed inputs. The 50-step latency results use
one timed observation per configuration on a shared 910B3. See the [test log](review-tests.log),
[pre-commit log](review-precommit.log) and [performance and quality measurements](summary.json).

At the time of this review, human review, DCO sign-off and upstream CI were
subsequent steps. The integration used MindIE-SD PR 929 at
`6985241966212ee36cc1f10a222e03e8154330e9`.

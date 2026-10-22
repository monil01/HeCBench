# Native066 prospective source-bound recovery

**Private proposal; no reserved raw directories, builds, targets, or GPU queries have been created/run by this preparation.** Exact future namespace is root-reserved `/tmp/hecbench-roofline-v100-20261020/batch066-rsc-secp-source-bound`. Root must review the packet, then CPU-stage with the provided helper before either collector call. Canonical C21 remains unchanged. No new master, no repeated039/041, no GPU5.

## Exact changes and limitations

- **RSC native** parent `batch003-rsc`: kernel and host source unchanged. Only existing compile definition `-DRAW_DIR="/home/imo/HeCBench_julia/.porting-state/porting_logs/rsc-julia-shared/raw/cuda"` moves to fresh066 scratch/private_capture. Macro appears only in output-directory creation and raw capture filename (study_cuda.cu:42–43); fixtures/oracle/model-fitting DSO/flags remain byte-identical. Source output-ofstream failure exits3, matching observed rc3 after kernels, but several other branches also exit3; failure cause is not localized, so recovery is unproven. Requested timeout remains original180s, independently applied to build and profile. All four RANSAC paths, property/oracle checks, raw captures, and five STUDY_TRIAL rows remain mandatory.
- **Secp256k1 native** parent `batch003-secp256k1`: actual build stopped wall_timeout180.59s and `nvcc: Terminated`; no source-rejection diagnostic. Compiler/source/flags and argv3 retain the full three512-point chains and five checked trials. Two previously implicit read-only fixture files are explicitly SHA-bound in the new plan (fixture98308B, expected_bytes147B). Timeout180→900 applies independently to **both build and profile**, since existing C21 has one CLI budget; this is not mislabeled as a build-only override. No alternative collector, math, or selector is introduced.

Both prospective JSON plans differ from their exact original parents only by fresh attempt prefix, RSC's one output macro, and the two explicit secp read-only asset bindings. Removing those two supplemental bindings, reversing that one macro and the attempt prefix recovers the entire old plan object exactly. `plan_source_proof.json` records these identities, all source/library/fixture SHA checks, selected config/result joins, and future launch argv. Source bytes are not copied or changed. Staged:true in the prospective plan bytes describes the required future staging state; no existing collector is pointed at this private proposal tree.

## Root CPU staging and launch

Create a root-owned review JSON with `approved:true` and `delivery_sha256` exactly matching this packet's DELIVERY.json. Then invoke the source-reviewed helper:

```sh
python3 /tmp/roofline-completion-20261021/rsc_secp066_recovery/stage_after_root_review.py --apply --root-review ROOT_REVIEW_JSON --root-review-sha256 ROOT_REVIEW_SHA
```

The helper revalidates every delivery, source, fixture, DSO, result and collector SHA before creating the fresh066 root. It never compiles or launches anything. It creates only private attempt directories and the exact approved plan bytes, journals failures without deleting any bytes, and rehashes inputs after staging. It refuses an existing066 root, foreign/symlinked raw root or changed binding. No extra asset payload copies are needed; inputs are read-only canonical paths. A failed/partial stage stays held for root review, not silently reused.

After independent source review and successful CPU staging, **root only** may run the exact two commands recorded in `plan_source_proof.json`: C21 collector with absolute source/plan root, `--benchmark rsc --timeout180` or `--benchmark secp256k1 --timeout900`, `--skip-qualified-selected --retain-auxiliary --reserve-mib2048`. All original default legacy2022 metrics, kernel replay/selector and Landlock/source/device checks are unchanged. This packet neither launches those commands nor authorizes any bypass. No discarded/shortened primary workload, newmethod adoption, or guaranteed recovery is claimed.

The first private preparation script had a Python whitespace/ternary syntax typo; its failed draft is retained. This is shared CPU plan-authoring overhead, not a target/compiler/JIT attempt.

# RBC fresh044 method-recovery proposal

CPU-only staging, no build/GPU/test. Fresh plan root:
`/tmp/hecbench-roofline-v100-20261020/batch044-rbc-retained-samples`.

The existing reviewed `stage_RBC_samples.py` generated all four slots unchanged. Bound copied header/controller hashes equal038; seven original GPU function definitions remain byte-identical. Original038 native failure (relative `tools/v100_profile_write_guard.py` resolved from scratch) and pending model history are archived in `preserved_batch038/`. This is an observer/launcher path failure, not a GPU numerical failure.

Source scope: full1000-step warmup and five full1000-step checks, retainedtrial2 zero-based step240 marker, Nu history1→2 and exact dispatch vector `[3,6,2,11,5,0,1]`. Source12167 cuBLAS API calls is not a GPU launch census. The first-per-mangled-specialization/geometry sampling is partial; no complete timestep aggregate, whole1000step roofline, historical timing replacement or causal bottleneck claim.

Future ROOT-ONLY invocation (not executed):

```sh
python3 /home/imo/HeCBench_julia/tools/v100_roofline_collect.py \
  --source-root /home/imo/HeCBench_julia \
  --plan-root /tmp/hecbench-roofline-v100-20261020/batch044-rbc-retained-samples \
  --profile-window-reviewed --skip-qualified-selected --timeout 1800 --retain-auxiliary
```

No selection-overrides flag: the collector supplies the exact reviewed selector and rejects a simultaneous override. Timeout/auxiliary retention are prospective resource proposals for root; preserve all source math, checks and state if resource choices change. The path-canonicalization fix is parent-owned. Parent reports adopted collector SHA `2d5355a75f2b8250b5cca21f2e637b0e268dd9098dd7e474f68c1392402871f9`; no worker edit or execution of collector. Parent must serialize with owned predecessors and preserve foreign-GPU protection/global lock/UUID/writeguard.

`recovery_review.json` binds all source/config/result/plan hashes and preserved038 files. `profile_window_review.json` is the exact collector-consumed review. Installed2022 CLI help is copied as method support, not proof that any marker/sample has executed. Runtime markers, full five checks, real counter identities and profiler ordinal behavior remain unverified until root collection. No allocation count or success inferred from source alone.

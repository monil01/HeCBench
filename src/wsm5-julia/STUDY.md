# WSM5: conditional full microphysics column restoration

The historical Julia file is a marker kernel that prints zero rain/snow. This study reconstructs the complete original 753-line water-vapor/cloud-liquid/cloud-ice/rain/snow operator, including thermodynamic feedback and sedimentation, rather than preserving that placeholder.

## Reference intervention and scientific scope

Original native `fallc_k` and `fallc_kp1` read themselves before initialization. Preserved dry-column arbitrary-storage completions produce 0 versus 0.005 mm precipitation despite no hydrometeors. The owned conditional native copy changes only their initialization to zero; original bytes and CPU warnings/witnesses remain. No unsafe original GPU execution was needed. This corrected-native stratum is explicitly excluded from untouched-native aggregates.

Named dry 9×7, warm 60×46 (original footprint), and cold 13×10 columns use three vertical levels, initial ice zero, uniform 500 m spacing, 10 s step and one sedimentation substep. All source phases/formulas remain; general initial-ice, variable-spacing, multi-substep policies and original all-0.001 CLI are unmeasured. This is not a complete atmospheric-model or climate-skill certificate.

## Numerical evidence

All four models pass five checked trials. Every retained artifact has 212480 Float64-encoded values: 103365 exact actual-device initial-field/header words, then all 15 complete final fields and 64-element prefix/suffix guards per buffer. Float32 scientific storage is exactly expanded into Float64 encoding. Readonly fields, dry manufactured identity, and guards are exact. Predeclared numerical tolerance is 1e-7 + 5e-5|reference|. A full corrected-source CPU oracle shares algorithm provenance; independent water-budget conservation, positivity, finite-state and snow-fraction checks provide additional witnesses, without independently rederiving every empirical rate. Saved GPU physics reviews cover all20 artifacts; no extra GPU timing series was collected.

Native-vs-Julia maximum absolute discrepancy is 3.0517578125e-5 (temperature-scale fields), accepted under the unchanged pre-GPU relative criterion. Near tiny hydrometeor values, relative errors can be large against the CPU oracle; absolute/relative budgets and all invariant checks remain explicit. CUDAjl/KA/JACC artifacts are identical for these inputs, not asserted universally.

## Mapping and effort

Full typed C-AST translation preserves source Float32/Float64 usual arithmetic, assignment narrowing, lexical scopes, x/k/y layout, 8×8 groups and all18 runtime bounds. Source double locals and unsuffixed constants entail actual Float64 work despite Float32 fields. Private MVector4 storage/local diagnostic initialization changes resource mapping; no ISA/register/bottleneck claim is made. CUDAjl uses its own CUDA.rsqrt API; KA/JACC own allocation/dispatch call CUDA.rsqrt explicitly, so both are hybrid_device_intrinsics with no cross-backend claim.

Two full generated reference bodies were CPU frontend-checked: the first failed C-shadowed-name declarations (B2), and alpha-renaming fixed it. The final body passed its first GPU JIT. One shared host spacing B2 preceded device JIT. KA passed first candidate/JIT. JACC flat34-argument forwarding failed InvalidIR (B4); packaging18 runtime bounds in a typed tuple, with no constant specialization or mathematical change, passed the second candidate/JIT. All failed candidates/logs remain. Historical costs are unknown; prospective nonexclusive cached/repeated-context activity counters and partial source-size proxies are separate.

## Timing

Five inprocess trials, explicit warmup outside, mean trials2–5. Sum of three actual kernel+context-completion ROIs; reset/initial D2H/output/CPU checks are outside. Means: native0.00014186925 s, CUDAjl0.00038922375 s, KA0.000378886 s, JACC0.000366883 s. These small pipeline times include host launch/completion effects; they are not GPU-event-only throughput or causal framework/roofline measurements. Required provenance is in SCOPE.md, fixture_manifest.json and the canonical configurations.

Final result: `.porting-state/portable_julia/runs/20261018T142408929003Z_wsm5/result.json`. Source/config frozen after verification; reproduction uses `python3 tools/verify_portable_julia.py --config src/wsm5-julia/study_config_jacc.json` through the global GPU6 lock.

# Linear regression: complete two-stage scoped study

Original sources are unchanged and archived under `reference_history/`. The source GPU-only path uses odd63-thread workgroups for both sufficient-statistic and centered-variance reductions, with CPU aggregation/coefficient feedback between them. All three Julia versions implement both genuine kernels with their own framework dispatch; no native kernel or vendor reduction substitutes for them. Nativefloat2/float4/float2 AoS strides remain component-fast in flattened Julia buffers.

## Scope

The original temperature file `../lr-sycl/assets/temperature.txt` is absent. Named deterministic datasets have63,126,96453rows (source full footprint), divisible by63. Per-row `x=(i%63-31)/16`; `y=1.25+.75*x+noise`, with noisezero for63 and `((i*17)%7-3)/64` otherwise. These are exact dyadicFloat32 values. Source repeat100000 becomes3/3/7; cpu_offset0 only. Arbitrarypartialworkgroups/CPUheterogeneousmode/originalCLI assets are unverified.

## Evidence and timing

Every trial saves all actualdeviceinputs (193291Float32 prefix words), all first/second-stage blockarrays, mean/intercept/slope/truncated explainedvariance report and384guardvalues/case, totaling203659words. Every input is rechecked read-only. Primarydual-reference tolerance1e-4absolute+1e-5relative was declared before runs; guards/inputbytes exact. IndependentFloat64OLS coefficients must be within1e-4, measured9.8117e-6. Float32odd-tree CPUreplay additionally checks every block; its integerpercent report allows one truncation unit while the actual captured four-modelpercent comparisons use the strict primary tolerance.

Five checked trials; report mean2–5. The ROI sums source repeated kernel loops plus context completion for both stages/allcases (26launches). Allocation, transfers, CPUsolve, checking and one-launch-per-stage/case warmup are outside. No full allocation-inclusive or original100000repeat timing claim.

KA first candidate's explicit return was rejected by the macro frontend before JIT; history retained. JACC.shared performs an extra252/126float scratchcopy and initialization barrier before source lane overwrites. This is an explicit work/mapping confound, not a general unsupported feature or pure compiler lowering comparison.

## Replay

No generated external data asset is needed: fixture formulas are in `study_support.h` and `study_common.jl`, bound in every config. Recorded pinned versions: Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0, CUDA12.0/sm70. Requiresnvcc/g++ and installed environments. Recorded absolutepaths in configs must be relocated for another host.

```sh
python3 tools/verify_portable_julia.py --config src/lr-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/lr-julia/study_config_refs.json
python3 tools/verify_portable_julia.py --config src/lr-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/lr-julia/study_config_jacc.json
```

For fresh replay, remove reusefields or point each nextconfig to its freshly generated priorresult; historical localrawresults may not exist in a checkout. Centralverifier serializes GPU6. Canonical local traits/status/effort plus tracked pervariant verification/accounting and paper_observations provide complete sourcebindings and prospective usage windows. No statistical modelquality or observed climate trend is established by syntheticfixtures.

# MT — original Mersenne Twister and Box–Muller pipeline

## Measured scope

The primary case preserves the original **4,096 streams**, **5,860 outputs per stream**, seed **777**, 32×128 launch geometry, 19-word private UInt32 state, complete tempering/uniform conversion, and the source Cartesian Box–Muller transformation. It runs three internal kernel-pair repeats instead of the original 1,000-repeat CLI; the original repetition count is unmeasured. Two additional checked controls use 38 outputs/seed 0 and 20 outputs/seed UInt32 maximum. Those controls do not enter the reported primary time.

All nine original source/build/asset-declaration files are byte-preserved in `original_native/`, with source notices retained. `study_native_driver.cu` includes the unchanged native GPU kernels and links the unchanged original CPU references. No native arithmetic repair was needed. Each Julia model has its own two kernel entries and dispatch; later versions reuse the validated Julia mathematical bodies and own Julia host orchestration.

The exact original DVC archive is 79,247 bytes, MD5 `2d20b003f0064b7fd8fed7cb394a955e`. It was recovered from an existing local readiness cache, not synthesized or downloaded during this task. `asset_provenance.json` records archive/file hashes and checks all 4,096 stripped GPU parameter records against the original full CPU records. The two required assets total 327,680 bytes, listed in `essential_assets_for_commit.json`.

## Numerical evidence

Final receipt: `.porting-state/portable_julia/runs/20261020T020635793735Z_mt/result.json`. Final configuration: `study_config_jacc.json`.

Every model has five checked trials after a full warmup. Every trial retains **48,676,885 Float32 primary values**, comprising:

- 196,629 exact input-prefix values: metadata (UInt32 seeds split into exact UInt16 halves) and every actual GPU parameter byte encoded as an exactly representable Float32 UInt8 value;
- all 24,240,128 actual final Gaussian outputs across the full-size case and both controls;
- all 24,240,128 uniform outputs from a separately identified, untimed, deterministic MT replay.

The timed final Gaussian array is copied **before** replay overwrites its buffer. The uniform values are not mislabeled as captured timed intermediates. Full exact independent recurrence checks establish their deterministic seed/parameter policy. All **20 complete primary artifacts are bit-identical**, including native CUDA. No canonicalization, approximation of UInt32 inputs through Float32, or statistical-summary substitution is used.

The independent CPU checker is an authored **block-twist** UInt32 recurrence (different organization from the incremental GPU recurrence), exact source Float32 uniform conversion, and LongDouble log/sqrt/sin/cos evaluation for every Box–Muller value. The predeclared pointwise budget is `32*2^-24*(1+r+abs(log(u1)))`, using the source Float32 angular constant and rounding. All **242,401,280 checks per model** pass across five trials. Native runs additionally pass the unchanged source CPU reference's `L1 < 1e-6` criterion in all 15 case/trial combinations. Prospective dual tolerances remain `atol=1e-5, rtol=2e-6`; observed artifacts are stronger, bit-identical evidence.

Every trial retains 3,072 actual prefix/suffix guard bytes with complete allocation headers and hashes. These guards are captured after the timed Gaussian work, before each auxiliary uniform replay. Small controls use the logical prefix of the full physical 96,010,240-byte output allocation; no separate small-logical-end or post-replay guard capture is claimed. Output completeness and the independent full recurrence checks remain explicit. Descriptive moments/ranges are recorded, **not** accepted as distribution-quality, period, independence, or Monte Carlo accuracy certificates.

## Timing and effort

Only the full-size, seed-777 source pair loop plus whole-context completion is timed, divided by its three internal repeats. Allocations, copies, CPU checks, guards, replay and control cases are outside this region, matching the source timing boundary. Discard only outer trial 1's time and average trials 2–5.

| Version | Mean seconds per full-size pair | Whole-program / GPU-JIT processes | Actual translation-window tokens |
|---|---:|---:|---:|
| CUDA | 0.003506354 | 1 GPU process; 1 build invocation | Native readiness separately: 2,037,836 |
| CUDA.jl | 0.004938418 | 1 / 1 | 1,588,482 |
| KA | 0.004964750 | 1 / 1 | 1,098,656 |
| JACC | 0.005076188 | 1 / 1 | 1,690,239 |

Each target's first complete GPU candidate passed. The CPU checker library is compiled separately once per target process; that is not a GPU candidate or backend compiler invocation. One JACC CPU evidence-audit command initially named the wrong immutable run folder and failed closed; its corrected path passed without any GPU rerun or criterion/source change. Historical IRIS results are contextual source/asset-readiness clues, not current CUDA/Julia verification, attempts or token costs.

Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0, StaticArrays 1.9.18, CUDA 12.0, V100/sm70. All models use framework device kernels. KA/JACC need no CUDA device intrinsic extension, atomics, shared memory or vendor RNG; their per-thread `MVector{19,UInt32}` is ordinary Julia private state. JACC explicitly uses `shmem_size=0` and fresh launch specs. Register/local-memory placement and SFU/bandwidth bottlenecks were not profiled, so the timing differences are not attributed causally to them or used for a roofline claim.

Tokens are prospective, context-inclusive activity counters, including repeated/cached context, verification and coordination. CUDA.jl's window includes common harness creation; KA/JACC inherit validated bodies and host logic. Native preparation and annotation are separate. Nested verification windows overlap translation and must not be added. These counters are not billing, exclusive coding costs, independent generation costs or recovered historical consumption. Backend compiler invocation counts remain unknown.

## Reproduction

Recorded configurations retain absolute local paths and source-bound reuse receipts. On the recorded host, run the central verifier sequentially:

```sh
python3 tools/verify_portable_julia.py --config src/mt-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/mt-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/mt-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/mt-julia/study_config_jacc.json
```

For a fresh/relocated execution, set source/executable/evidence paths and remove `reuse_result`/`reuse_models`, preserving inputs/criteria/timing scope. Existing prerequisites are nvcc, g++, pinned Julia project environments and the two parameter assets. Use the runner's serialized GPU6 lock on the shared study host. No privileged commands or operating-system/package changes are required.

CPU-only regeneration from the exact archive and complete evidence auditing:

```sh
python3 src/mt-julia/prepare_runtime_assets.py --archive /path/to/data.tar.gz --output-dir /tmp/mt-data
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/mt-julia/audit_complete_evidence.py --result .porting-state/portable_julia/runs/20261020T020635793735Z_mt/result.json --model jacc --folder .porting-state/portable_julia/mt_primary --output /tmp/mt-audit.json
```

The auditor requires an existing NumPy installation (recorded 2.2.6); replace the illustrative `PYTHONPATH` with the evaluator's environment. It fails closed on missing/truncated files, scans complete arrays and rehashes every primary artifact. Keep raw captures, full DVC archive, executables and checker `.so` out of Git/Overleaf. Source-bound scientific and implementation traits, actual effort windows and RQ1–5 observations accompany the final receipt.

# Paged-attention: named full FP32 computational pipeline

The original CUDA source and all historical/failed candidate copies remain preserved. This study runs the original FP32 uniform/normal initialization and paged-attention computational bodies, with declared controlled inputs, equal query/KV heads, block16, 128threads, head sizes32/64/128 and full lengths/tails/repeated-page/ALiBi/constant-value checks. It is not original fullCLI/allprecision or end-to-end LLM certification. Original BF16 actually aborts on V100sm70; missing target BF16 execution is a native hardware boundary, not KA/JACC unsupported evidence.

The source CPU checker reads packed keys as dense. `packed_layout_witness.json` and its tracked CUDA witness demonstrate the discrepancy; unchanged native GPU math agrees with an independent dense packed-layout oracle. Source max-shifted softmax divides by sumexp+1e-6, deliberately retained. See reference_provenance.json. No nativeGPU body repair.

## Regenerate exact fixtures on the recorded environment

From repository root, CUDA12.0 nvcc/V100sm70:

```sh
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 nvcc -O3 -std=c++17 -arch=sm_70 src/attention-paged-julia/generate_fixture.cu -o /tmp/attention_paged_fixture
cd src/attention-paged-julia
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/attention_paged_fixture
sha256sum fixture.bin oracle.bin
```

Compare every byte count and SHA256 in `data_regeneration.json`. The generator actually executes source GPU initialization and independently computes the Float64 dense oracle; large regenerable arrays are not committed. Different device/compiler math may produce different bytes and is a different fixture, not recovered historical evidence. Original headers, source and small scientific witnesses are tracked. No external data dependency.

From repository root after regeneration:

```sh
python3 tools/verify_portable_julia.py --config src/attention-paged-julia/study_config.json
python3 src/attention-paged-julia/audit_outputs.py RESULT.json
```

The central verifier owns the GPU6 lock, builds native source in isolation and executes allfour fresh on clean checkout. Relocate the explicit paths in a local config if checkout differs. Use pinned Julia1.12.6 `_julia_ka_env`/`_julia_jacc_env`, CUDA6.2.1/KA0.9.45/JACC1.4.0/StaticArrays1.9.18 and localCUDA12.0. Five checked inprocess trials with one prior compile warmup; mean trials2–5. ROI is sum3attention launches+context completion; generation/overrides/allocation/readbacks/math checks outside. Complete generated inputs, actual device input bits, all outputs and32guards are captured. The external independent gate runs before each accepted trial row. `audit_outputs.py` is CPU-only and rehashes every saved array, recomputing independentmath for each distinct byte-identical complete record.

CUDAjl/KA/JACC own dispatch, shared storage and barriers, but use explicit CUDAwarp32shuffle/fastmath backendextensions; no tested crossvendor support claim. KA index rematerialization witness and immutablevalue mapping are recorded. JACC.shared adds zero initialization+barrier with explicit416/864/1632sharedbytes, no extra global inputbuffer. No measuredcounter/bottleneck/roofline claim.

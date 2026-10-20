# Bounded asynchronous word2vec workflow property study

The additional all-four workflow property study is complete for the named bounded corpus. Historical CBOW component evidence and the two original native CLI repeats are preserved under the parent directory.

This additional study runs the complete source host vocabulary/Huffman/unigram construction, 20 pthread sentence/subsampling controllers, 15 epochs, original GPU CBOW and memset kernels, and original binary vector export. It uses a prospectively named single-line cyclic corpus (524288 tokens,1024 types), dimension128/window8/negative25/sample1e-4. It does not certify text8, dimension200, model identity, generalization, production training quality, or all CLI branches.

The source's plain GPU Hogwild updates and shared host sentence/alpha races remain. Observers capture actual GPU-consumed inputs and per-position RNG states. No SGD serialization is added. Complete models and output files remain raw; the central exact artifact compares only the lossless static-input certificate. Every trial independently checks complete models/traces, guards, a fresh deterministic unit-gradient control and a prospectively fixed full expected-NCE diagnostic. Dynamic schedules, consumed IDs/alphas and RNG draw counts are endogenous observations, not falsely identical inputs.

Physical sentence buffers are explicitly zero-initialized; this is a conditional initialization/integration intervention. Instrumented GPU bodies differ in telemetry, with original mathematical expressions preserved. CPU reentrancy frees and fresh state resets support warmup plus five trials. Allocation/initialization/work/export are in the ROI; explicit observation readbacks and CPU checks are excluded. These timings are excluded from untouched-source causal rankings.

Regenerate owned source/fixture: `python3 src/word2vec-julia/async_workflow/prepare_native.py`. Build with `nvcc -O3 -std=c++17 -arch=sm_70 -Xcompiler=-pthread -I src/word2vec-cuda src/word2vec-julia/async_workflow/study_native.cu -o /tmp/w2vfull/native` under the shared GPU lock. Generate config with `make_native_config.py`; launch `run_study.py --config .../config_native01.json` (central runner owns the lock). Existing NumPy is required on PYTHONPATH. No system installation required.

Callback bound: a full call contains six1024-word sentences, consuming at least6144 source words; per-thread epoch quota26214 is tested only after calls. At most five full calls plus one conservative partial EOF/newline allowance perthread/epoch gives1800; trace cap2048 aborts any overflow. Raw table400MB is retained once, with actual complete device byte comparisons for every run and exact monotonic-table RLE certification. Prospective total new captures budget<1.9GB/tmp; no old evidence removed. Local/tmp evidence must be preserved before reboot.

## Final evidence and reproduction

Final result: `.porting-state/portable_julia/runs/20261020T155223860367Z_word2vec/result.json`; all four property paths pass. `full_saved_property_audit.json` recomputes all20 complete properties without modifying captures;415 raw files and73 execution-source bindings rehashed. `source_binding_audit.json` binds exact measured dependencies. All20 raw syn0 models are distinct. Primary CBOW component results/costs remain unchanged; this additional configuration has role `workflow_property_secondary` and contributes no second population name.

**Repair history:** `study_bridge.cu`/`libworkflow.so` and earlier CUDAjl/KA runs retain a per-request synchronous pointer fetch; those timings are excluded by the parent hash-bound `study_adjudications.json`. Selected `study_bridge_v2.cu` caches the allocated trace-counter pointer on the host before training and performs no CUDA call in the queue broker. No Julia mathematical kernel changes were required. Rebuild selected provider under shared lock:

```
nvcc -O3 -std=c++17 -arch=sm_70 -Xcompiler=-pthread -Xcompiler=-fPIC -shared -I src/word2vec-cuda src/word2vec-julia/async_workflow/study_bridge_v2.cu -o /tmp/w2vfull/libworkflow_v2.so
```

Use existing `numpy==2.5.4` (parent `cpu_requirements.txt`); set PYTHONPATH to that installation and OPENBLAS_NUM_THREADS=1. The source-owned launcher currently selects the existing user-space installation `/tmp/pasc-julia-plot-tools`; adapt that explicit local path for relocation. No OS installation is needed. Julia packages use the repository-pinned KA/JACC environments. Generate each clean target config with `make_target_config.py --model MODEL --reuse-result PRIOR/result.json --output OWNED/config_MODEL.json`, then the owned central launcher; order native→CUDAjl→KA→JACC. `make_native_config.py` rebases native paths. Preserve the existing corpus/table before reruns; mismatched canonical bytes fail rather than silently replace data.

CPU-only recheck: `PYTHONPATH=/path/to/numpy OPENBLAS_NUM_THREADS=1 python3 src/word2vec-julia/async_workflow/audit_saved.py --result /path/to/result.json --output /tmp/word2vec-audit.json`. Raw referenced artifacts must exist; hashes alone cannot substitute for them. Dynamic libraries/native binary and generated corpus are regenerable and excluded from source Git. `/tmp` captures require archival before reboot. No evidence was deleted.

# Fresh guarded one-update TSNE verification

This strengthens explicit source-buffer guard evidence for the existing **revised, one-update** contract. It does not certify the failed strict three-update contract, the original1000-update trajectory or embedding quality. Earlier captures/configs/math/metrics remain unchanged. New final receipt: `.porting-state/portable_julia/runs/20261020T151316499476Z_tsne/result.json`; new configuration `tsne_mnist60000_K32_I1_revised_fourier_units_explicit_source_buffer_redzones_v5`.

All four models execute a full untimed warmup and five checked trials; mean trials2–5: CUDA0.05601287625s, CUDA.jl0.8815865105s, KA0.91097261175s, JACC0.95342451125s. Keep these **new guarded timings separate** from old unguarded measurements. Guard storage/allocation/initialization and required completion/free remain inside the full-pipeline ROI; added guard readback/check/serialization time is excluded. Existing allocation/reduction/host/synchronization differences remain covariates; no pure-frontend or causal speedup claim.

## Complete evidence

`full35_saved_audit.json` replays all20 artifacts (2,605,107,760B) and all35 fields against unchanged independent Float64/Complex128 formulas, with127 strict source/dependency bindings. Maximum field error/budget ratio0.01509198793, using the original fixed budget. The old four-field lossless power-of-two codec remains unchanged; all original raw GPU bits recover exactly. Inputs and categories remain exact. This is not a newly relaxed tolerance or a checksum substitute.

`guard_saved_audit.json` independently rechecks432,640 actual head/tail bytes across765 source-buffer allocations (39native/trial,38Julia/trial). Every native source `thrust::device_vector` allocation across all active source translation units uses `guard_allocator.h`; source32controller vectors plus7perplexity helper vectors. Every Julia explicit source NamedTuple array uses a borrowed original-length payload within a guarded allocation. Both ends have64 elements; evidence retains all actual guard bytes and expected values before free. Different allocation counts reflect preexisting helper-buffer layout adaptation, not omitted guard checks. The native observed controller and all three model entry files are byte-identical to their original delivered counterparts. Original native GPU source files and `../pointwise.jl` are unchanged; only observer/allocation adapters and timing subtraction are new.

**Coverage boundary:** explicit source GPU arrays only. CUDA/Thrust/KA/JACC internal reduction/broadcast temporary pools and CPU memory are not guarded. Canaries detect modified head/tail writes, not every out-of-bounds read or general memory safety. CPU counterfactual one-byte checker rejection is reported as a CPU gate sensitivity check, not a fault injected into a GPU execution.

Original strict I3 integer-grid failures, earlier I1 raw pointwise Fourier rejection and I1000 no full-trajectory certificate remain preserved. Perplexity50 remains infeasible for31 nonself neighbors; source search returns31, and source normalization/application quality limits remain. See `../../_portable_study/stencil_ports_537_audit_20261020/full_verification_worklist.md` for the unclosed scientific tasks. No new full-source/scientific-quality headline follows from clean guards.

## Reproduction

Requires the existing pinned blocking environments and original TSNE data/oracles; see parent README for exact cold asset/oracle regeneration, original kernels and dependencies. No OS/package/privileged installation is part of this work. NumPy2.5.4 is already available through `/tmp/pasc-julia-plot-tools` in this recorded environment; change only an owned future launcher when relocating Python dependencies. Existing source/root data paths in archived configs are immutable.

From the repository root:

```sh
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/tsne-julia/guarded_I1/build_native.py --output /tmp/tsne-guarded-I1-native
python3 src/tsne-julia/guarded_I1/make_config.py --models cuda --output src/tsne-julia/guarded_I1/configs/reproduction_native.json
python3 src/tsne-julia/guarded_I1/run_study.py --config src/tsne-julia/guarded_I1/configs/reproduction_native.json
```

Then generate/run new distinct configs sequentially with `--models cuda,cuda_jl`, `cuda,cuda_jl,ka`, `cuda,cuda_jl,ka,jacc`, and `--reuse-result PREVIOUS_RESULT`. The existing verifier owns the shared GPU6 lock; do not outer-lock `run_study.py`. All4 measurements are fresh for the guard configuration; only already accepted models of this **same new configuration** are reused while completing the next target.

After final execution, `PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/tsne-julia/audit_saved.py --result FINAL_RESULT --oracle /tmp/tsne_oracle_mnist_i1 --output /tmp/guarded_full35_audit.json` and `python3 src/tsne-julia/guarded_I1/audit_guards.py --result FINAL_RESULT --output /tmp/guarded_bytes_audit.json` replay all numerical/raw-codec and guard evidence without GPU execution.

Storage: approved4GB/tmp transient peak,2.606GB final captures, existing432MBoracle reused, no old capture deletion/move and no large home writes. Sidecars remain local/tmp at exact result-row paths and are hash-bound in the audits; preserve them before cleanup/reboot. No binaries are essential nonregenerable assets: build recipe and frozen source are retained.

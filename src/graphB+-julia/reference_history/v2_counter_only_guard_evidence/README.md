# graphB+ four-model study

This study translates the complete active 16-entry signed-graph balancing pipeline. The original Texas State BSD sources are unchanged and archived under `original_native/`. CUDA.jl, KA and JACC execute their own kernel entry points; all three explicitly depend on custom LLVM volatile helpers, and KA/JACC also depend on CUDA warp and atomic APIs. No Julia target calls a native device kernel.

## Verified scope and reference qualification

Three connected synthetic graphs have 33, 65 and 129 nodes, with one, three and three source iterations. Their maximum degrees 32, 64 and 128 exercise both warp partition branches. Signs include positive, frustrated and initially balanced graphs. The original native CSV reader matches the fixture CSR exactly. Selected roots have unique degrees, avoiding an unspecified degree-tie order.

The full pipeline includes hashed BFS parent selection, bottom-up subtree counts, top-down interval labels and warp CSR partition, tree-edge counting, cycle-sign balancing, positive connected components, largest-component selection, quotient Bellman–Ford and parity voting. An independent CPU oracle checks signed Int32 hash comparisons, deterministic maximum-parent ties, all subtree intervals, sign potentials, components, distances and complete directed-edge/node counts. Every trial retains 35,841 Int64 values, including 2,856 actual post-initialization input values. Exact comparison uses zero absolute and relative tolerance.

The original native five-trial series passed. A faithful CUDA.jl version exposed 19 reverse tree-edge initialization differences between timed execution and replay. The original `initMinus` has conflicting writes across blocks. These observed reverse edges are not printed by the source CSV output, so this is **not an observed wrong native CLI result**. The separately qualified corrected-native secondary assigns each edge one owner in `study_corrected_init.cuh`. The other 15 native bodies, one initialization launch and the strict oracle remain unchanged. This configuration is excluded from untouched-native aggregates. Source, original passing captures and rejected target captures remain separate.

## Timing and raw evidence

Each model has a full warmup and five checked trials. Only trial 1's timing is discarded; the mean uses trials 2–5. The ROI sums the three source pipelines, 156 launches, host tail/convergence reads and whole-device completion. Allocation, reset, source initial `init`, input audits, independent checks, serialization and tagged replay are outside. It is a preallocated pipeline timing, not an allocation/file-I/O-inclusive CLI timing.

Actual timed final arrays equal the final replay arrays. DFS labels are explicitly supplemental **untimed replay** labels; no timed intermediate capture is claimed. Complete raw CSR, EdgeInfo and worklist buffers are retained locally. The CPU audit checks 200 raw files, 89,120 directed slots and complete per-row bijections. Atomic worklist order is normalized with multiplicities retained; literal raw order identity is false. All allocation guards are checked by the producing process, with zero bad-byte counts saved; individual raw guard bytes are not retained.

The source's original 792-node dataset was subsequently recovered from the exact declared DVC object: 29,537 archive bytes, MD5 `6e794be5f5af14cd6f104f3a44a7219b`. Its 10,127 undirected edges form one component. This recovery is readiness evidence only: the original dataset and 10,000-iteration CLI were not run in the named verified configuration. No social-network ground-truth or arbitrary-graph quality claim follows.

## Reproduce

From the repository root:

```sh
python3 src/graphB+-julia/generate_fixtures.py
g++ -O3 -std=c++17 -shared -fPIC src/graphB+-julia/study_check.cpp -o src/graphB+-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/graphB+-julia/study_config_native_corrected.json
python3 tools/verify_portable_julia.py --config src/graphB+-julia/study_config_refs_corrected.json
python3 tools/verify_portable_julia.py --config src/graphB+-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/graphB+-julia/study_config_jacc.json
```

Configs record local absolute paths and prior result reuse. On a fresh host, relocate paths, rebuild the checker, and remove reuse fields or point them to freshly verified corresponding results. GPU 6 on this host is serialized by the central runner; do not add an outer lock. Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0, CUDA 12.0 and V100 sm70 are the recorded environment.

The root `study_config_native.json` and `study_config_refs.json` describe the **historical original-source attempts**, not the corrected study. Their exact measured source versions are archived under `reference_history/`; do not replay those configurations against the changed current source and call that original evidence.

The essential generated fixture is `fixtures.bin` (168,984 bytes, SHA256 `1740ea598f7afe1dd07c354198b4e7be44c90af4312284a08f4b5457265eca40`). Generated `study_check.so` is local, not a source asset. `asset_provenance.json`, `initialization_race_witness.json`, `raw_state_audit_all4.json`, per-model receipts and `paper_observations.json` document sources, feedback, exact checks, token windows and scientific limits. Source helper reuse and context-inclusive activity costs are explicit; these are not independent first-principles or billed translation costs.

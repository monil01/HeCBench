# HBC four-model study

Full original exact/default Brandes kernel and bitonic helper, original80CTAs×1024threads. The original recovered gene-coexpression graph is included, plus path33/grid35/star1057 controls. Optional approximation is unmeasured. Original native bytes/license are archived under `original_native/`; no mathematical native repair.

## Assets and reproduction

Run from repository root. Installed Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0/Atomix, CUDA12.0 nvcc, g++17 and an sm70 V100 were used. Pinned environments are `src/_julia_ka_env` and `src/_julia_jacc_env`. No OS package changes were made.

`dataset_readiness/provenance.json` records exact DVC object848186B/MD5`150f618aa33568b1b8c1c857b6e8a3ac`; extraction yields original `breast_coexpress.txt`. `asset_provenance.json` hashes fixtures/oracle/generators/originalsources. Essential small binary assets are `fixtures.bin`3189012B and `oracle.bin`143120B; recovered archive and textgraph are readiness/provenance assets, not raw measured output.

```
python3 src/hbc-julia/generate_inputs.py
g++ -O3 -std=c++17 src/hbc-julia/study_oracle.cpp -o /tmp/hbc_oracle
/tmp/hbc_oracle /home/imo/HeCBench_julia/src/hbc-julia
g++ -O3 -std=c++17 -shared -fPIC src/hbc-julia/study_check.cpp -o src/hbc-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/hbc-julia/study_config_jacc.json
python3 src/hbc-julia/audit_full_states.py --result .porting-state/portable_julia/runs/20261019T213352051148Z_hbc/result.json --output /tmp/hbc_raw_audit.json
```

The generated `.so` and native binaries are rebuildable and must not be committed. To execute on a relocated checkout, regenerate absolute config paths/reuse links, compile checker/native with `-DHBC_ROOT='"/absolute/new/src/hbc-julia"'`, choose an explicitly authorized GPU, and collect fresh measurements; local archived raw references are not bundled as if portable. Acceptance files remain the pre-GPU declared contract; oracle stats generation time may change, never rewrite archived runconfig hashes.

Each model warms the complete four-case pipeline before five reset checked trials; report mean2–5. The ROI is sum4kernel launches+whole-contextcompletion; allocations/reset/uploads/readback/guards/fullworkspaces/oracle/divideby2 are outside. Original allocation-inclusive outer CLI time is not compared. CUDA.jl maxregs64 and pinned-JACC wrapper maxregs64 preserve source1024threads; rejected defaults90/98 registers are archived. JACC static scratch/compiler-control are explicit CUDA-backend interoperability, not portable JACC claims. KA uses own localmem/barrier and Atomix.

`all4_raw_audit.json` binds final result, all20trial artifacts and80rawcases/actualguardfiles. Dynamic CTA source assignments and unread workspace tails are retained, not forced to rawidentity. `paper_observations.*`, scientific-domain snippet and per-model accounting state exact scientific and performance limitations. No biological validation or unmeasured bottleneck claim.

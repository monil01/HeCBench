# SSSP four-way study

## Verified scope

Full NYR graph (264,346 vertices, 733,846 positive weighted directed edges), source0; original hybrid threshold128 and supplemental GPU-only threshold0, plus weighted/disconnected and narrow-chain controls. All4 models execute five full convergences per configuration after untimed warmup; mean trials2–5. Complete unique Int32 distance vectors agree exactly with both GPU references and independent heap-Dijkstra/Bellman/tight-predecessor checks. All80 full observable artifacts and80 raw private arrays were audited; no checksum-only acceptance or queue canonicalization.

Original actual Makefile R100/warm10 CLI and R1/l0 CLI passed independently. BAY dataset, other graph sizes, negative cycles and arbitrary queue capacity remain unmeasured. Geographic units/application routing quality are not certified.

## Implementation and timing

Every Julia target authors the one GPU workgroup-queue kernel. Original CPU seed and one-worker phases are held common through `host_bridge.cpp` + original CPU-only `kernel.cu`; the bridge contains no custom GPU kernel/runtime import. Whole pipeline: hybrid host control; GPU component: framework device kernel. CUDA.jl uses own static shared/atomics, KA uses public localmem/barriers+Atomix, JACC uses own shared/sync_workgroup+Atomix. The latter sync API is unexported/version-bound; foreign CUDA device calls are absent.

Source logical shared queue1600Int32+3scalars =6412B. JACC selected mapping additionally preloads a1603-wordzero seed and synchronizes per block; actual full seed+17guards captured at process boundaries. All streams explicitly legacy and whole-device completed. Atomix1.2.1 CUDA RMW maps to relaxed/device-scope CUDA atomics even when monotonic requested; source-bound package review retained.

ROI sums completed CPU seed/worker and GPU compute segments for one convergence, excluding graph/frontier transfers, controller/readbacks/checks/output. Five retained source work counts may differ because concurrent atomic relaxation changes frontier order and hybrid partition. Timings are not uninterrupted CLI walltime, kernel-only throughput, or a causal frontend comparison. NYR retained means seconds: native0.02810457375, CUDA.jl0.03191361525, KA0.0364430045, JACC0.05699092225.

## Qualified reference and scientific boundaries

Original GPU body and CPU arithmetic unchanged. Defined otherwise-unused full initial queues, external guards, and packed scalar storage are explicitly reference/integration adapted (`native_reference_repaired:true`, excluded from untouched-native aggregates); original NYR distances were correct. No known scientific native defect is asserted. Original source mixes plain loads/atomic updates and bounded local1600 queue; observed zero overflow/full distance certificates do not establish formal memory-model safety or all-graphs correctness.

Actual first-GPU complete input graph/cost/color/queues/scalars+guards are serialized; hybrid final observable may finish on CPU. Raw last-GPU state and full private host/device arrays retained separately without falsely calling intermediate GPU costs final. GPU-only configurations require actual final GPU costs equal Dijkstra. The necessary private-cost bound is not an independently reconstructed valid-walk certificate. Not every overwritten per-level frontier is retained.

## Cold assets and evidence

`regenerate_assets.py --output DIR` retrieves the exact declared DVC archive when needed (MD5 e26e35f389a5f6119caad624ead9e473), verifies it, regenerates graph/oracle binaries and CPU helper. Optional `--archive /path/data.tar.gz` avoids network. Cold CPU regeneration produced all7assets9,103,500bytes exactly; no irreplaceable input. See generator `--help` for proof option and `fixture_manifest.json`/`cold_regeneration_proof.json` for hashes. Do not commit raw GPU outputs/shared binaries or force-add regenerable large assets.

`prepare_cold_configs.py --output CONFIG_DIR --capture-root OUTPUT_DIR` prepares fresh no-reuse four-model configs against supplied source root; current-root CPU preparation was checked, not a fresh relocated GPU execution. Execute via `PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/sssp-julia/run_study.py --config CONFIG` under the existing pinned projects. Launcher sets only its subprocess STATE under/tmp and preserves canonical unique-run symlinks and storage receipts; existing captures are not moved/deleted. Preserve local/tmp evidence before cleanup/reboot. GPU6 global lock remains central verifier-owned.

Final results and complete source/asset/package bindings are listed in `final_source_binding_audit.json`; detailed80-array+80-private audit in `final_*_full_array_audit.json`. No accepted trials rerun. Token counters in `observed_tokens.json` are actual context/cache-inclusive disjoint activity windows, not exclusive billed translation. Prior strategy reuse and JACC cross-benchmark SPH review make all3 clean-inclusive-design calibration ineligible. Supplemental SPH metadata cost is recorded separately in SPH; not retroactively reassigned here.

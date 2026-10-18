# Exact shared-atomic scatter aggregation study

Run `python3 tools/verify_portable_julia.py --config src/scatterAdd-julia/study_config.json` from the repository root. The central runner locks physical GPU6 and compares five complete outputs against CUDA and CUDA.jl. `make run` executes the named reference.

Both original native shared-aggregation kernels are included unchanged. They retain512threads,8192-source-row chunks, sourcefloor(N/8192)+1grid including empty tail blocks, shared atomics then global atomics. Kernel1 has an extra initialization barrier; kernel2 simplifies address loops. Five named shapes cover batch/thread/chunk and vector/warp boundaries and a single-destination contention case. They have at most three blocks and do not establish saturated throughput or certify the original ten-million-row CLI.

Each timed path executes three accumulating launches from the same nonzero initial output, reset once outside the clock. The expected result is init+3*indexed_sum. Signed integer/16 contributions and initial values obey a proven absolute-partial-unit bound below2^24, so every possible Float32 shared/global atomic addition order is exact. Independent integer histograms, original CPU math, component conservation, full outputs,170guards and exact input-prefix bytes are checked. This zero-tolerance contract is a justified subset, not a claim about general non-dyadic rounding or invalid indices. Intermediate launch states are not captured.

CUDA.jl and KA use typed Int32 index staging. JACC.shared(vector) always starts at dynamic-shared offset0, so two independent calls would alias. The JACC port uses one Float32 buffer and numerically stages bounded indices0..31 exactly before conversion back to Int32. This supports indices below2^24, explicitly scoped here; it adds a full template copy/barrier. Nominal shared capacity matches the source; KA static buffer padding and actual registers/traffic are unprofiled.

All ten configurations warm before five outer trials. Only trial1 timing is excluded. Allocation, nonzero reset, transfers, CPU math and artifact construction are outside synchronized launch/completion timing. All samples remain without outlier removal. Source-reviewed traits and timing data do not establish a causal bottleneck.

The historical Julia marker/PASS placeholder and original Makefile remain in reference_history. The source comment references embedding-style T5 token compression, but no whole-model or compression-quality claim is made. No native benchmark-local license file was present; no license is invented.

Essential binaries: fixture.bin and expected_outputs.bin, pinned by fixture_description.json. independent_artifact_audit.py verifies all20 saved captures without another GPU run. Source hashes, provenance, per-model ontology/failure events and prospective effort windows are separate from unknown historical translation cost.

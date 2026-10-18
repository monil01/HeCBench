# Wallace identical-stream study

Run the central verifier from the repository root: `python3 tools/verify_portable_julia.py --config src/rng-wallace-julia/study_config.json`. It serializes GPU 6, reuses unchanged hash-checked reference/KA captures and checks all five target trials against both references. No timing series is replaced.

`make run` here executes the named CUDA.jl study under the same GPU lock. The archived original CLI is separate and remains unverified at its default 16,384-block regime.

This study launches 257 blocks of 256 threads, with unchanged 2,048-value source pools and one output/pass per kernel. Three chains use seeds 1, 0x12345678 and 0xffffffff for 1, 7 and 31 updates. Three internal repetitions each start from the same captured state; resets/transfers/CPU checks are outside the timed chains. Five outer trials are retained and only the last four times averaged.

The source emits the incoming pool times chi-square scaling and then updates the pool. Full emitted and updated arrays are both checked against original CPU Float32 and independent Float64 permutation/orthogonal-matrix oracles, with guards and energy invariants. This is deterministic finite-stream equivalence, not a certification of RNG distribution, independence, period or production statistical quality.

Essential binaries are `fixture.bin`, `source_cpu_outputs.bin`, `expected_outputs.bin`; their pre-run hashes are in `fixture_description.json`. The fixture generator is host-only C++ using the original KISS/Box–Muller and chi-square helpers. The independent oracle/audit use NumPy 2.5.4. `independent_artifact_audit.py` reads all saved artifacts without GPU execution and includes an energy-preserving wrong-state counterfactual. Original authors are retained in unchanged native headers. `reference_history` preserves original Julia source/Makefile.

# Six-path convolution study

Configuration: `resnet6_source_tiles_defined_halo_v1`.

Run `python3 tools/verify_portable_julia.py --config src/resnet-kernels-julia/study_config.json` from the repository root. The central runner serializes physical GPU 6 and checks every target against CUDA and CUDA.jl. It reuses unchanged, hash-checked reference and KA captures already recorded in the configuration. New independent measurement requires an explicitly named configuration, not deleting prior evidence.

`make run` in this directory executes the named CUDA.jl study with GPU 6 and the common lock. It accepts a repetition count, not the original mode/repetition CLI. The original Julia file and Makefile are preserved in `reference_history/`; they were an elementwise placeholder and are not a valid convolution reference.

The independent fixture generator requires NumPy 2.5.4. Captured fixture and expected-output binaries are essential study inputs; their hashes and all fixed sizes are in `fixture_description.json`. `input_bounds_witness.py` proves the original native allocation defect. `reference_correction_provenance.json` was declared before first measurement. CUDA kernels are unchanged; the physical input tail is initialized explicitly, and results are excluded from untouched-native aggregates.

`independent_artifact_audit.py` independently checks all 20 saved full artifacts and exact prefix/border/guard bytes. Its run ID is explicit; it reads immutable artifacts and does not execute the GPU. `path_timing_*.json` retains all five samples and means of the last four for every path. `study_adjudications.json` adds only hash-bound scope/provenance context, without modifying raw executions or tolerances.

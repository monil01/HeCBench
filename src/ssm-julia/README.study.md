# Gated serial state-space recurrence

Run `python3 tools/verify_portable_julia.py --config src/ssm-julia/study_config.json` from the repository root. The runner serializes GPU 6 and verifies all five captures against CUDA and CUDA.jl. `make run` runs the named Julia reference under the same lock.

The selected source serial-base kernel evolves eight hidden components per batch/channel through every time position, with softplus-positive step sizes and SiLU gating. Three named shapes retain sequence lengths1/33/1024 and channel counts1/33/257. Each shape has one-call and three-call paths; state continues between calls, and every call writes a separate captured output buffer. Initial hidden states are nonzero. Every y value and final state is independently checked, including exact input prefixes and17guards per buffer. Hidden states at intermediate time positions are not GPU-captured.

The original Julia GPU kernel is preserved byte-identically, with only an autorun guard. Its historical checker printed fixed PASS messages and mislabeled the serial kernel as a parallel VLLM scan; those messages are not evidence. The original main and Makefile remain in reference_history. The current study explicitly verifies serial-base only. Original VLLM CUB scan, full default CLI, alternative flags and model task accuracy remain unverified, not framework unsupported.

The source full-file build fails C++17 parsing at an unselected VLLM templated lambda. native_base_source.cuh extracts the original prefix through the serial kernel, changing only the relative reference.h include. The selected GPU body is byte-identical and no native algorithm or arithmetic flags are repaired. Full original CLI compilation is not certified.

All six paths warm before five outer trials. Each trial times one complete chain per path; only trial1 timing is omitted from the mean. State reset, buffer allocation, H2D/D2H, independent CPU math and artifacts are outside synchronized launch/completion timing. This is not the historical mixed reset/allocation timing and not a parallel scan throughput claim. Raw timings and accepted numerical drift are retained without outlier removal.

Essential binaries: fixture.bin and expected_outputs.bin; fixture_description.json pins hashes and shapes. independent_artifact_audit.py verifies existing saved artifacts without new GPU work. Its CPU counterfactual demonstrates that resetting state between calls changes later outputs. Independent analytic channels cover exact unchanged state and homogeneous exponential decay. NativeCPUFloat32 and independentFloat64 recurrence checks are distinct.

No benchmark-local license file was present in native source; preserve original source attribution comment without inventing a license. No profiler counters, saturation claim or causal bottleneck attribution is made.

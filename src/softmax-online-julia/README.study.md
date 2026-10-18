# Block-online softmax study

Run `python3 tools/verify_portable_julia.py --config src/softmax-online-julia/study_config.json` from the repository root. The runner serializes GPU 6 and checks all five complete captures against CUDA and CUDA.jl, reusing hash-checked references. `make run` executes the named Julia reference.

This study preserves the original native kernel4 block-online algorithm. It covers all six original block sizes (32 through1024) and five shapes, including three rows of50257columns. It does not certify native kernels1–3, the default8192-row CLI or model-level accuracy. The historical genuine Julia three-pass implementation and Makefile are archived; current Julia reference maps the same online algorithm as native.

Independent Float64 normalization checks every probability, plus finite/nonnegative bounds, row mass, bit-exact ±1000 paired-row shift invariance and untouched guards. All20 artifacts are bit-identical and retain exact serialized inputs. Finite inputs only. Paired rows are controlled invariance cases rather than independent data.

Each outer trial sums30 separately timed paths; each path averages20 kernel launches. All configurations are explicitly warmed before the five trials. Only trial1 timing is excluded; trials2–5 are averaged without outlier removal. Allocation, reset, H2D/D2H and CPU checks are outside synchronized launch/completion clocks. The aggregate is not best-block-size or best-native-algorithm throughput.

All models use4096bytes shared reduction storage at every width. JACC additionally copies a shared template and adds a barrier; actual compiled instructions/registers/traffic are unprofiled. The wide-row case has only three blocks and is not saturated throughput. Do not infer a roofline bottleneck or causal abstraction overhead from these timings alone.

Essential binaries are fixture.bin and expected_outputs.bin; hashes and case definitions are in fixture_description.json. The independent artifact audit uses saved captures, not another GPU series. Its recorded paths are machine-local. Source digests, scope adjudications, per-path timings and effort windows are preserved separately.

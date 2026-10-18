# Sigmoid loss operator study

Run `python3 tools/verify_portable_julia.py --config src/scel-julia/study_config.json` from the repository root. It serializes physical GPU 6 and reuses unchanged hash-checked reference/KA captures. Every target has five full-array checks against CUDA and CUDA.jl, with only the first timing excluded.

`make run` here executes the named CUDA.jl study under the common lock. Original Julia/Makefile are archived in `reference_history`; they were a touch-kernel placeholder, not a numerical cross-entropy reference.

All three original source formulas are tested as distinct operators. Standard BCE uses softplus(l)−l*t; log-D uses (2*t−1)*softplus(−l); unjoined uses (1−t)*softplus(l)−l*t. Only standard BCE with labels in [0,1] carries its usual nonnegativity interpretation. The source-like Gaussian+1 target case is unbounded and is checked algebraically.

Five shapes exercise inner lengths 1,257,513,1025,4097, extreme/signed-zero logits and reduction tails. Twenty launches/path are averaged within each of five trials. Output guards reset outside clocks, all outputs/inputs transferred and checked outside clocks. Native original `-use_fast_math` is retained; the Julia implementations use the same manual 256-lane tree and default CUDA math. Native uses CUB BlockReduce, with internal resources unprofiled.

Essential binaries `fixture.bin` and `expected_outputs.bin` are hash-bound by `fixture_description.json`. `generate_study_fixture.py` uses only the Python standard library. The independent full-artifact audit also evaluates a separate cancellation-aware Float64 oracle, without changing any measured execution or tolerance. The original PyTorch/Caffe2 license is preserved. These are forward operator checks, not gradient, trained-model or default 2048×50176 CLI accuracy claims.

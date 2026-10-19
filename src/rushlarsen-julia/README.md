# Rush–Larsen four-model study

The original CUDA sources are unchanged; exact copies are under `reference_history/`. All three Julia launchers execute the full 19-state generalized Rush–Larsen cell update in `study_device.jl`, retaining source SoA, old-state caches, Float64 expressions and 256-thread blocks. KA/JACC dispatch their own framework kernels. No native device kernel is called by a Julia version.

## Named scope and verification

Three cases: 33 cells/one zero-time step; 257 heterogeneous cells/100 steps; original Makefile 100000 cells/100 steps. The default million-step CLI is unmeasured. Each of five trials checks all actual device input bytes, all 19 final states, untimed deterministic first-step replays, both allocation guards, read-only parameters, source CPU outputs and 13 independent affine gate/receptor closed forms. Absolute/relative full-state bounds are 1e-12/1e-10; independent affine-gate absolute bound is 2e-13, declared before GPU measurement. The ROI sums three source time-step loops and context completion (201 launches); initialization, one-launch-per-case compile warmup, capture and CPU checks are outside. Trial1 discarded from mean, but checked.

Original CPU/GPU and an independent high-precision derivative witness refuted an early hand-parenthesis analysis error. This is preserved as shared analysis feedback, not a source defect. JACC candidate1 reused its mutable LaunchSpec across sizes; the pinned backend cached one block after the 33-cell warmup. A diagnostic confirms missing tail at257. Candidate2 supplies explicit per-size block count. Math/tolerances are unchanged. The failed process log/source survive; its unreported temporary binary does not.

## Rebuild fixtures and mathematical body

Run from repository root, with Python3, g++ and the source tree present:

```sh
python3 src/rushlarsen-julia/study_fixture.py
g++ -O3 -std=c++17 -x c++ src/rushlarsen-julia/study_cpu_oracle.cpp src/rushlarsen-cuda/reference.cu src/rushlarsen-cuda/utils.cu -o /tmp/rush_oracle
/tmp/rush_oracle "$PWD/src/rushlarsen-julia"
python3 src/rushlarsen-julia/study_translate_ast.py
```

The AST generator requires pycparser3.11. The recorded environment reuses `/tmp/wsm5-python-tools`; elsewhere install pycparser3.11 into the active Python environment. Generated body is already committed, so regeneration is unnecessary for execution. `asset_provenance.json` records exact generator outputs and immutable source archives. Only the three small fixture/oracle `.bin` files are source assets; trial captures are local evidence, not source files.

## Execute

The committed study configs document recorded absolute tool/source paths. Relocate those paths when running on another host; Julia1.12.6, nvcc12.0, g++, CUDA6.2.1, KA0.9.45 and JACC1.4.0 were used. GPU6 on this host is serialized by the central verifier. Run native, refs, KA, JACC configs in order; fresh replay requires removing reuse fields or updating them to the corresponding newly generated result (never silently reuse unavailable local historical outputs).

```sh
python3 tools/verify_portable_julia.py --config src/rushlarsen-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/rushlarsen-julia/study_config_refs.json
python3 tools/verify_portable_julia.py --config src/rushlarsen-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/rushlarsen-julia/study_config_jacc.json
```

See `paper_observations.json`, per-model `verification.json`/`translation_accounting.json` and canonical local traits/status files for numerical evidence, effort units, actual observed token windows and scope limitations. No long-horizon/clinical validation or cross-backend portability is claimed.

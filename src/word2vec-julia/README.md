# word2vec: source CBOW kernel component

## Scope and source integrity

This is a **named kernel-component contract**, not a complete port/certificate of original multi-threaded corpus training. Original `cbow.cu` is included unchanged by `study_native.cu`; CUDA.jl, KA and JACC implement both actual GPU definitions (zero-init and CBOW) with own framework device APIs. Source host vocabulary construction, corpus reading, asynchronous sharedbuffers and embedding-file output are not restored by this driver. No missing original text8 corpus/DVC object was substituted without disclosure.

Two actual original CLI runs used a deterministic generated12×1024-token corpus, size128/window1/negative1/hostthreads1/oneepoch. Both exit0; complete128-dimensional vectors differ maxabsolute0.000138. Crossblock unsynchronized read/modify/write and host sharedbuffers are static source evidence, but exact contribution to that observed repeatdifference is not isolated. This is not a claim of a framework limitation or scientifically unintended Hogwild algorithm.

Selected source-kernel scope:

1. All1024 positions, source1025blocks×128threads, one cold-start sentence,200features aligned224. Disjoint positive/negative targets and zero output weights make the initial output-weight gradient independently tractable. Input embeddings remain unchanged, while395,474 output-weight values distinguish the result from a no-op. Identical zero updates still share input rows: no general race-freedom claim.
2. Three original one-block/position hot-state controls, each6sentences: lookup/dim128, positive clipping/dim64, negative clipping/dim200. These exercise nonzero hidden errors and embedding updates but do not certify hot full1024-position throughput.

Original generic GPU body branches are translated, but window1/negative1, named8192-entry unigramtable and finite controls do not certify all CLI modes, zero-target fallback/skip branches, arbitrary shapes or training trajectories. cbow/hs flags do not select alternate actualGPUalgorithms in this source.

## Numerical evidence

Final result `.porting-state/portable_julia/runs/20261020T073115229620Z_word2vec/result.json`. Each model has full untimed warmup and five actual checked trials; discard timing1 only. All20captures contain5,377,048Float64-encoded words (43,016,384B each). This losslessly represents finiteFloat32/32bitinteger actual data; it does not change GPU precision. Exact2,688,524-word device input prefix includes all physical arrays and constant-table readback. Independent Float64 context means/dots/SGD oracle uses source lookup approximation, not the128-lane reduction implementation. Predeclared atol1e-6,rtol2e-5; RNG/inputs/17tailwords×5arrays×4cases and inactive rowpadding are exact. The original fixed1000-element constant symbol has complete readback but no appended redzone.

`final_saved_full_audit.json` rechecks all20 full files/48fields against oracle and22declared source bindings/8source trees, with nontrivial no-op rejection in everycase. Original failed native attempt is preserved: CPU oracle RNG/sentence output order was wrong; the serialization correction changed no GPU arithmetic/tolerance. JACC host LaunchSpec constructor attempt failed pre-device-JIT; archived candidate uses exact source, corrected factory is launch_spec.

## Timings and implementation boundaries

Mean trials2–5 in seconds: CUDA0.000105426, CUDA.jl0.00020880425, KA0.0005048345, JACC0.000364948. ROI is four CBOW kernel launches plus wholedevicecompletion, with initialization/reset/transfers/readbacks/validation outside. Zero-init source kernel is actually executed and checked outsideROI. All three classes are framework_device_kernel for this measured boundary; ordinary host launch/completion does not imply hybridhostcontrol.

Julia versions use extra fullgroup barriers to implement original volatile warp-lockstep dependencies. CUDA.jl/JACC use dynamic sharedmemory; KA uses compile-time localmemory, capacities1024/1536/2304B. JACC own shared API also preloads a zero template before mathematical work. Sourceconstant expTable is a readonlyglobalargument in Julia. These ordering/resource/hostcall differences preclude a pure frontend or causalperformance claim. No measured counters/roofline evidence.

## Reproduction and local assets

PythonNumPy2.5.4 is recorded in cpu_requirements.txt. Existing projects pin CUDA6.2.1, KA0.9.45, JACC1.4.0. Fresh verification environment probes observed localruntime12.0 and compilerartifact12.9; native is nvcc12.0/sm70. These freshprobes are not original trialstdout. No package/system/OS changes or sudo invoked.

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/word2vec-julia/regenerate_assets.py --output /tmp/word2vec-cold/assets --proof /tmp/word2vec-cold/proof.json
python3 src/word2vec-julia/build_native.py --output /tmp/word2vec-cold/native
python3 src/word2vec-julia/prepare_cold_configs.py --assets /tmp/word2vec-cold/assets --native-build /tmp/word2vec-cold/native --output /tmp/word2vec-cold/config
```

Actual cold CPU regeneration reproduced fixture.bin+oracle.f64 (53,770,628B) exactly, rebuilt native and prepared a fresh all-four config; **no extra cold GPU run** is claimed. Large generated binaries and rawGPUcaptures stay local/tmp; deterministic recipes/hash inventory are committed essential assets. Source original corpus diagnostic generator is described exactly in native_diagnostics/source_review.json.

## Effort and remaining requirements

CUDA.jl/KA one program candidate/check each; JACC two candidates/evaluations but one actual device-JIT process, one hostAPI repair, zero GPU math repairs. Target prospective context-inclusive counters864,367/775,221/1,706,636tokens; nativeprep2,550,536 and verification1,691,728 are separate. Cached/reasoning counters are subsets, not extra billable totals. Limited component scope is calibration-ineligible for complete corpus translation cost. Full async training/output agreement, original corpus quality and broader parameter coverage remain explicitly unverified.

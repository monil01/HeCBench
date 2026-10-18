# Minisweep: complete manufactured transport sweep

## Operator and scientific scope

The source miniapp has three face-boundary initializers and a persistent wavefront sweep. Each(octant,energy)worker traverses every spatial front/cell, applies dense moment→angle transforms, weighted upwind face recurrence, angle→moment transforms and Float64 global atomic accumulation. Allfour original native kernels are unchanged, and allfour genuine kernels/callbacks are authored for CUDA.jl, KA and JACC. Historical Julia touched an array and printed hard-coded norm/diff/PASS/GFLOPS; reference_original preserves that stub. Its matching default norm was not execution evidence.

This is a deliberately manufactured transport-style operator. Matrix rows annihilate affine moment/angle components; powers-of-two spatial/energy/unknown scaling and specially selected face weights produce the source exactidentity. The affine property is in the moment/angle basis, not a spatially affine analytic PDEsolution. No physical cross-section/conservation/positivity/productionreactor accuracy or convergence is certified.

## Independent full-output verification

Case1 retains actual source default5×5×5,30energygroups,33angles,NU4,NM4. The complete60000-value state is exactly invariant and has normsq6895251000. Case2 uses2×3×2,2energygroups,5angles and non-affine moment perturbations; its actual output differs from input by up to461.7198982, so an identity/touch shortcut cannot pass.

generate_oracle.py independently uses CPUlexicographic topological upwind traversal and NumPy2.5.4 small BLAS products, distinct from GPUwavefront loops. Full three output states, three facefields and persistent angularscratch are checked. Every model runs five checked in-process trials after warmup; mean2–5. Each artifact has3419552Float64values:62848actualinput/parameter prefix exports,3356192complete outputs and512deviceguards. All20artifactbytes are identical and every CPUoutputerror is actuallyzero, within unchanged predeclared1e−8+1e−10absref. The independent audit covers120complete iterationstates and160face/scratch tensors, exact inputbytes/guards and sourcebindings. Relative error denominatorfloor1e−12 is explicit.

## Actual source scheduling and iteration semantics

Source main fixes oneZblock, nproc1 and one step with all eightoctants active. The supplied --nblock_zflag is not consumed; internal MPI/multi-Zblock variants and the larger32×32×64/E64/A32 run are unmeasured, not unsupported.

R3 repeats use unchanged deviceinput. Source CPUhostpointerexchange never updates d_vi. Untimed state DtoH after every iteration is retained and allthreeactualstates captured. Source-like last-two-readback comparison can differ from immutableinitial-input comparison; iteration_semantics_observation.json derives both from savedactualstates without claiming originalCLIexecution on our custom fixture. These are repeated fixed-input operators, not physical time evolution or convergence steps.

## Runtime mapping and performance

Source faceblock(8,8,4), waveblock(4,64) and wavegrid(2,1) are retained. There are240 validpersistentworkers in the default case and16 in the supplemental case. Each worker serially executes its source loops; octant/energy facefields are private and outputmoments use relaxed device-scope atomic adds. No authored blockbarriers orsharedmemory are needed. Source/APIreview confirms Atomix1.2.1modify! onCUDA uses native Float64 atomicadd onsm70 and ignores macrorequestedordering; JACC1.4exports that macro. This is an official portable atomic API, so allthree model classes are framework_device_kernel, with concrete backenddependencies disclosed.

ROI is full outputzeroing+threefaceinitializers+persistentwavefront+wholedevicecompletion, averagedR3 percase and summed overtwo cases. Interiterationreadbacks are outsideROI as in sourcektime; original overall hosttime is not compared. Native cudaMemset versus Julia ordered-taskstream cuMemsetD8Async is an explicit API/streamcovariate. Mean total milliseconds:CUDA33.413475, CUDA.jl39.098263, KA40.378934, JACC39.172270. Componenttimings retains both case means and everyoutertrial.

Source index/math/launch mapping is retained, including Int32 helperindices and explicit FMA transforms. Actual codegeneration/registers/occupancy/noalias/bounds policies and traffic are unprofiled. Small grid is a source parallelism property, not a measured bottleneck or roofline point. Hostdispatch/descriptor/resetAPI and compiler differences prevent causal abstraction-only conclusions.

## Attempts, cost and reproduction

Current CUDA.jl reference has one authored algorithm body with four kernels, two processchecks: reservedidentifierlocal parse rejection before GPUJIT, then successful renameangular. That is one hostfrontend repair, one actual device-JITprocess and zero devicecompiler/numericalfeedback repairs. Proactive restoration from the old stub and two CPUoracle protocol revisions are separate shared effort. Nativebuild1; KA/JACC each firstcandidate/processcheck/fourkernels passed, zero feedbackrepairs. Kernel/grid/iteration/trial counts are not attempts.

Historical usage is unknown. Saved-source proxies and mechanically shared reference math/driver are separate from marginal targetentries; prospective cumulative context/cache/coordination windows overlap workflow and are not exclusive billing. Four essential CPU input/oracle binaries are reproducible from generate_oracle.py and fixture_provenance.json; case1 oracle is about26.8MB. reference_history retains the unmeasured initial oracle protocol/provenance and hostparse failure. GPUoutputs remain local by hash. LICENSE.source preserves the original notice.

study_config.json selects20261018T084135383541Z_minisweep. Root alone commits sharedsource and each verified variant separately; canonicalguides and workercheckpoint resume the queue.

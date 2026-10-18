# streamCreateCopyDestroy: real runtime copy/stream lifecycle, zero authored kernels

Native CUDA, CUDA.jl, KA and JACC each pass five complete captures: 51574089 UInt32 words/trial,32809-word actual hostinput rawIEEE32 prefix, all12202destination buffers with4096values+128guards each, and actualsuccessfulcopy/create/destroy/synchronize counters. Independent pre-GPU binary32 oracle/golden checks every saved byte; no checksum-only result. Full sourcehostinputs verified unchanged, source1.618f+i preserved.

## Scope and ontology

Eight named diagonal cases use original test0/5/10/15 in both baseline/explicit-stream modes: streams1/2/4/8, buffers1/100/1000/5000, original iterations128/32/8/2. Other24 crossproduct cases unmeasured, notunsupported. Primary hardware/data-movement runtime study, no physical application. No authored kernels or device arithmetic in native/any target; device candidates0 is real absence, not unattempted work. Float32hostfixtureinitializer outsideROI; UInt32artifactencoding not deviceprecision.

Allthree wholeversions library_interop: own CUDA.jl/KA/JACC allocation outsideROI then shared explicit rawCUDART hostcalls forpageableH2D/defaultcontextcompletion or streamcreate/copy/sync/destroy. No fake @cuda/numericalkernel. JACCinitialization outsideROI completed before runtimework. Native actual runtimecallordering faithfully re-expressed in ownedharness to capture otherwise privatefreedbuffers; original source unchanged, no mathreferencecorrection. Extra successfulAPIcounterinstrumentation matchedall4; guards shiftpayload256B. Original hostoversizedunread newfloat[nBytes] tailnotclaimedcopied.

## Timing

Primary time sum8 rawelapsed caseROIs, means trials2–5 afterfullwarmup. Copybytes/iterationorder sourcepreserved. Baseline contextcompletion eachiteration; explicitmode creates defaultflagstreams eachiteration, stream-major/buffer-majorcopies then sync/destroys eachstream. Allocation/reset/readback/oracle outsideROI. Separate percase/source-normalized seconds periteration/stream are reported; aggregate not onecopylatency or bandwidth. Pageable staging/CPUcontention/runorder uncontrolled and variation not evidence for pure abstraction-lowering penalty. Source simultaneous samebuffer writes use identicalpayload, not genericracefreedom for differing data.

## Effort

Historical Julia print-only timing0/noGPU placeholder is source-reviewed S1, not an invented observedGPUfailure. Current native firstbuild/processpasses; initial sharedJuliahostcomprehension B2parserfailure before runtime repairedspacing. CUDAjl hostprocesschecks2, KA/JACC1each, frontendwrappers1each, authoreddevicecandidates/deviceJITprocesses0; actualbackendcompilerinvocationsunknown. Sharedpipeline conditionalreuse exposed, not independentkerneltranslation. Oldoriginalcostunknown. Prospective cached/repeated-context windows include science/source/metadata/verificationwaiting/coordination andoverlapsharedworkflow; no exclusivecoding/billedcost. Brief initialnative/legacyreadbeforeBEGINuncaptured andnotbackfilled. Mainchar4sourceproxy omits sharedruntimecode andnotconsumedtotal.

Final proof `.porting-state/portable_julia/runs/20261018T131405543293Z_streamCreateCopyDestroy/result.json`; canonical study_config_jacc.json. Means2–5(seconds): {'cuda': 1.196032395, 'cuda_jl': 1.03986446025, 'ka': 0.8099428367500001, 'jacc': 0.9165673910000001}. Sources frozen.

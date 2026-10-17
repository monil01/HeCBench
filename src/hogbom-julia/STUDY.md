# hogbom: scientific and implementation observation

Radioimaging CLEAN is a host-controlled iterativepipeline. Each iteration copies128devicepeakcandidates to CPU, resolves an absolute maximum, updateshostmodel andlaunches croppedPSF subtraction. Timing includes synchronization,D2H,managedhostallocation/control anddispatch, so no device-onlyrooflineclaim. The257wide equal-amplitudecase explicitlytests actualGPUhierarchical tiepriority: pixel32769 precedes300, unlikeglobalfirst. Fullmodel/residual/all19peakpairs arechecked and independent dirty=residual+shiftedPSF*model identity verifiesdeconvolution bookkeeping. CenteredunitPSF leavesnativeoffcenter yoffsetflaw inactive; everycapturedpeak proves thresholdstoppingdifferenceinactive. Results do not certify arbitraryPSF/CLI modes, numericalconvergence or astronomicaldatafidelity.

## Reference audit

Historical hogbom main only includes mcpr-julia/main.jl, a structural smoke dependency with no CLEAN argmax/PSF/model computation. Preserve original include and dependency hash; implement actualnative-equivalent CLEAN.

## Numerical and performance evidence

Configuration `hogbom_centered_psf65_257_19iterations_full_state_trace_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T232949185345Z_hogbom/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl repair andKA one candidate/JITprocess each. JACC first used JACC.Array (Base.Array) instead lowercase JACC.array, producing hostvectors and B4 non-bitstype GPUargument error beforelaunch; second candidate fixes onlyallocationAPI. Noobserved numerical/runtime repair. Prospectivewindows include samebenchmark verification and dp/fsm ownershipcoordination, not exclusivecodingcost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/hogbom-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Independent mathematical witness and scope

All20savedcaptures werechecked using `study_witness.py` from theiractualserializedinputs. Full model/residual/19peak traces match independent flatargmax+shiftedPSF Float32 recurrence exactly. Float64 reconstruction `dirty = residual + shiftedPSF * model` has maximumerror 1.1920929e-07. Every measuredpeak has absolutevalue at least 1.5496819, above1e-5, so nativeGPUlog-without-break versusCPUthresholdbreak is inactive. In the257case, actualGPUpriority chooses32769before300; globalfirstwouldchange the19iterationmodelby 0.154968262. This is a declared finiteiteration tiepolicy, not assumedCPUglobalfirst.

A static offcenter5x5witness withPSFpeak(1,3) gives nativefirstoverlapPSFy=-2 because `diffy=ry-px`. This documents an unmeasured scientific-contract limitation; the measuredcenteredPSF(3,3) does not trigger it. The offcenter corrected configuration remains a separatelyidentifiedbacklog. Native kernels were not altered; measuredcenteredresults are notclassified as demonstratedscientificfailure. No claim of CLEANconvergence or arbitraryPSF/CLIvalidity.

Timing is sum of two whole19iterationpipelines, including eachblocking2048Bpeakcopy, CPUargmax/hostmodelupdate anddiagnostictracewrites. Julia creates managedhost128Peak arrays, nativeuses stackarrays. JACC copies3072Bsharedtemplates with an extra initialbarrier. These are documented mapping/boundarycosts; no unmeasured bottleneck attribution.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

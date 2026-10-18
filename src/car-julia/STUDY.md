# CAR: content-adaptive image resampling

## Operator and declared fixture

Each output pixel/channel sums nine spatially varying weights times four-corner software-bilinear image samples with horizontal/vertical offsets. Named fixture B2/C3, non-square37×29input→23×19output, full2,622outputs, sourcepadding1/offsetunit1, R100. Positive spatially varying normalizedweights and signedquarteroffsets exercise allfourclampedges; exactdyadicimage is a batch/channel-dependent bilinearpolynomial. Float64closedform evaluates that polynomial at sourceeffectiveclampedneighborcoordinates, independently of the four-corner implementation. Source edge behavior clamps the floored neighbor then choosesR=L+1 while retaining originalfraction; no substitution with idealcontinuous-coordinateclamp.

Predeclared atol5e−5/rtol2e−6; all20completeoutputs pass CPUclosedform and all5dualGPUreference checks, all30,448actualinput/parameterprefixvalues and160guards exactserializedbytes. Partition error≤1.87e−8 and convexhullbounds are supplementary; the wrong historical kernel also passed the hull property. No trainedmodel or image-quality claim; originallarge/squareCLI is not certified by this cache-resident smallfixture. Generator uses only Pythonstdlib, preserves allinputbits and hash/provenance; essentialCPUfixtures are `study_inputs_f32.bin` and `oracle_f64.bin`.

## Correlated checker counterexample

Historical CUDA.jl was a genuine resampler, but `idx_img` usedimageheight as rowstride in bothCPUandGPU. Native usesimagewidth; square1024×1024default hides it. Namednon-square run actually failedallfivefullGPUoutputs (maxabs2.485149); accepted0/rejected5 artifacts remain in041719231496Z_car. Applying the historical CPUreference to the sameinputs/savedGPUoutput gives **RMSE0** and wouldPASS its1e−3threshold, while independently wrong by2.485149. `reference_selfcheck_witness.{jl,json}` preserves the actual CPUcounterexample; no new GPUJIT or inventedoldCLIexecution. Minimalone-line rowstrideW correction restores native semantics; native source unchanged and original reference source retained.

## Compiler loop and accounting

Two authored devicecandidates/threeprogramcandidate revisions/three process checks: initialhostSubArray→CuArraycopy fellbacktoscalarsetindex before authoredkernelJIT; collectCPUview fixes driver. Olddevicekernel then compiles but numerically fails; rowstride correction compiles and passes. Thus two real authored-deviceJIT checks, one numerical repair and one shared hostinitialization repair, not three device compiler failures. A separate CPU-only correlated-checker diagnostic is not a translation attempt. KA and JACC each onecandidate/check, no repairs, reusecorrectedpuremath/hostfixture already audited. Historical attempts/tokens unknown; prospective context/cache-inclusive activity windows and partialsavedsource proxies remain separate from exclusivebilledtranslationcost.

## Implementation and performance

All three variants are framework_device_kernel: independentlinearworker/output, 256threads×11blocks, no communication/sharedmemory; flatNCHWimage andB9HWweight/offsetplanes. CUDA.jl uses ownindices, KA globalindices, JACCparallel_for (explicitshmem_size0/syncfalse); commonFloat32math and JuliaInt64indexing retained. Native localindex/dimensionarithmetic usesInt32; indexing/hostlaunch/codegeneration remain covariates, no unmeasured cause asserted. Data cache regime is small; no inferred DRAM roofline/bandwidth attribution.

Each model warms the complete kernel beforefivecheckedinprocessoutertrials, each mean of100launches withblockingwholedevicecompletion. Allocations/copies/observations outsideROI. Report mean outertrials2–5. Actualseconds: {"cuda": 7.77922e-06, "cuda_jl": 1.43444125e-05, "ka": 1.58802975e-05, "jacc": 1.63128525e-05}. Originalnativefive-series reused unchanged, no fastestselection or tolerancechange.

Finalfour-modelbundle `.porting-state/portable_julia/runs/20261018T042111545633Z_car/result.json`; `audit_saved_outputs.py` confirms savedbytes/outputs and `source_integrity_review.json` checks actualmeasureddependencies unchanged. Reproduce with `make run`, Julia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,V100sm70. Parent coordinator owns separate commits/manuscript integration; this sourceobservation is paper-ready.

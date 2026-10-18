# Myocyte: corrected finite 91-equation RHS component

## Scientific contract
Actual cardiac ECC and three CaM/CaN/CaMKII compartment evaluators return all91 raw derivatives, three communication fluxes and91 host-coupled derivatives. Four probes comprise supplied initial state/actual parameter-reader semantics and explicit voltage, calcium, CaM, K/Mg and pacing/time perturbations. Source GPU int time truncation is preserved (0,6,5,1006). Supplied params.txt has16values, native requests18 and ignores EOF, retaining last value .57 in final slots; unchanged-reader CPUwitness proves this. No guessed physiological defaults.
This is component validation, not a validated cardiac trajectory or clinical prediction. All original13-stage adaptive hostsolver source was restored through common CABI, but its strict xmax1/2 diagnostic failed and remains unverified. Source default100ms not certified.

## Native failure and corrected secondary
kernel_cam.cu writes communication flux to d_finavalu[com_offset] rather than d_com. This conflicts with ECC writes to derivatives0..2 and leaves d_com unwritten for hostcoupling. Actual guarded native and faithfulJulia finiteprobes retained com12345.5 and failed full1440field captures. Static concurrent-write hazard proven; no isolated nondeterministic scheduling experiment claimed.
Owned-copy minimal destination-pointer change d_finavalu→d_com defines corrected_reference_secondary only. Original source/failures remain. Originalfullsolver rawstage nonfinite failure is not promoted or attributed solely to this pointer; original master .0001 NaN/Inf compensation does not validate a trajectory. MalformedInfJSON prevented full failed-trajectory artifact retention; logs persist, unknown rawhistory not invented.
See native_quality_witness.json, native_corrected/communication_pointer.patch, trajectory_scope_witness.json and reference_history.

## Complete numerical evidence
Final .porting-state/portable_julia/runs/20261018T091116492412Z_myocyte/result.json:20 full1440F64-encoded artifacts:444 byteexact serialized actual inputs,740 raw/com/coupled fields,256 tailguards. Allfive trial verdicts and dualrefs pass under unchanged predeclared2e-4+2e-4|reference|. Native/Julia maxdifference2.3473057808587328e-6; Julia versions mutuallybitidentical on these probes.
CPUFloat64 source-formula evaluation is a precision/transcription check, not independently derived physiology. Independently recomputed saved-field controls:240 volume-weighted CaM diffusion cancellation checks,320 exactzero derivatives and6320 sourceunchanged coupling slots. These do not imply global biochemical conservation or physiological trajectory validity.
All20 saved files and executed dependency/DSO/fixture hashes rechecked. Saved_output_audit.json records actual CPUerrors and bounds, no zero placeholders. One essential CPUgenerated fixture study_probes_f32.bin with generate_fixtures.py/fixture_provenance.json; GPU outputs local only.

## Implementation and performance scope
Native2blocks×32threads retained, two active workers: ECC group0/lane0 and serial threeCaM calls group1/lane0. CUDA.jl own indices, KA local/group indices, JACC index1–64 mapping and explicit shmem_size0. Allframework_device_kernel for measured GPUscope; original hostcoupling via CABI is outsideROI, not authored GPUlibrary delegation.
Timing is source GPUevaluator launch+wholedevicecompletion, mean20calls/probe summedfourprobes. Transfers/hostcoupling/oracles/allocations excluded; notwholeapplication orpureinstructionlatency. Explicit JIT/branchwarmup, fivecheckedtrials, discardfirst/meanlastfour, retainalloutliers. No measured roofline or guessed occupancy.

## Actual compiler loop and cost
Reference: historicalsurrogate restoration separatelycosted; five program revisions, ten externalJuliaprocesschecks (onehostparse, ninereachedGPUcheckpath), twodeviceIRfeedbackrepairs and one nativepointerrepair propagated toreference. Internal compilerpassesunknown. Own measurement/setup errors archived: badarchspelling, absoluteassetcopy, nvcc-Wl flag, datacountguard and omittedstructured:true yielding25records/model. Entire25series excludedbyprotocol, freshsolefive run, no fastestselection.
Native eightbuildcommands include twostandalonediagnosticbuilds; fourhostbuildscripts×twoDSOcompiles plusonepostverificationCPUreaderwitnessbuild. KA/JACC each onecandidate/devicecheck/zero repairs, sharing corrected reference/equations; notglobalfirst-programsuccess.
Prospective reference/KA/JACC/workflow/annotation windows are context/cache/coordination inclusive, not exclusive billing; historicalcostunknown, savedsourceproxies separate.

Mean seconds (sumfourmean20evaluations): {"cuda": 0.000129623, "cuda_jl": 9.9923675e-05, "ka": 0.00010928722500000001, "jacc": 0.0001264571}

## Separately excluded postverification diagnostic
One original-kernel/fullsolver diagnostic with defined12345.5 guarded communication buffers retained47,627 actualGPUfields locally (full_solver_capture_provenance.json). Forxmax1/2, status0 despite742/1631 nonfinite rawRHS fields,5nonfinite stageinputfields each, and750/1639 hostcompensations. All156/351 communication fields remain sentinel. This is not untouchedCLI uninitializedallocation or retroactive captureofoldprocess. One additionalnativebuild/GPUdiagnosticprocess belongs toannotation; acceptedtimingseries/criterion unchanged. Fulltrajectory unverified, causesnotisolated.

# spm: scientific and implementation observation

Native one-thread/grid-stride sampling: fixed97 Float32 jitter; actual fractional-z formula; affine map; bounds test; paired eight-corner trilinear interpolation; floor(value+.5) UInt8 quantization and validity mask. 128threads ×256blocks; each logical thread processes i then i+32768 until v³; dimensions17/33 include partial grid and second iteration; immutable97tuple in Julia replaces native local97array representation; PTX-aligned explicit fused arithmetic. Original strict nativeGPU/nativeCPU checker rejected1byte threshold drift (015105774443Z); allrawcaptures retained. Auxiliary criterion explicitly revised after observation to independent barycentric Float64 error interval gamma8*(250+.5)+32eps64*251=0.00011944776687047794. Exact unique rounded bin outside interval; only admissible adjacent bins inside. Native-vs-allJulia GPU comparisons retain atol=rtol=0 and masks exact. Initial actualGPU reference mismatch4bytes caught despite ownPASS, then NVCC PTX contraction order aligned without changing native source. Sourcecontract remains resampling only; no training/convergence/registrationqualityclaim. Full245100outputvalues and204308inputprefix values, four cases.

## Reference audit

Historical CUDA.jl actually implements source interpolation/gridstride, not a smokestub. Preservedoriginalbytes; hostFloat32vector jitter made immutabletuple fordeviceconstants; explicitFFMA order matched unchangednativePTX after4 exact-byte mismatches. OriginalCLI warmup/measurement and histogramonlychecker replaced by declaredstudyharness, notwholeCLIcertification.

## Numerical and performance evidence

Configuration `spm_two_shapes_identity_translated_fixedjitter_fullsamples_interval_v2`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T015648491177Z_spm/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Shared firstnative strictCPU-vs-GPU auxiliaryreject(N1) and Julia harness parserB2 (noGPUJIT). Secondsharedrun ownchecksPASSbut4exactGPUbytesmismatch(N2); explicitFFMAorder fix led exactagreement. Two referencekernelcandidates/threeprogramchecks(twoactualdeviceJITprocesses); KA/JACC onecandidate/checkeach. Nativekernelneveredited. Metadata/oracle work reused inorderedKAthenJACCworkflow. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/spm-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Independent quantization interval

For the selected Float32 coordinates, fractional weights and their complements are exact. All input bytes are at most250 and interpolation terms are nonnegative. A path has at most6 rounded stages plus final addition0.5; gamma8 conservatively bounds its error by gamma8*(250+0.5), with gamma8=8u/(1−8u),u=2^−24, plus32eps64*251 for the independent double evaluation. The bound is0.00011944776687047794. Only32 sampled values per capture have a half-bin threshold inside this interval; all others require the unique exact rounded answer. Complete corners/weights and admissible bins are in independent_witness.json. Example index29251,v33,g: exact-double interpolant207.50000381469727, native207, ordinary double-nearest208. Exact native/CUDA.jl/KA/JACC outputs still match without any output tolerance. No blanket one-byte acceptance.

All20 captures independently checked245100 outputs and204308 input-prefix values. Originalstrictauxiliaryrejection and lateractual4bytecross-modelmismatch remain in immutable runs; neither is relabeled as a compiler failure. Timed scientific region remains paired sampling, not registration optimization or histogram formation.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

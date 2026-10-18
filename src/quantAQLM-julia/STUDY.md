# QuantAQLM: exact half-precision quantized GEMV, corrected host extent/stride

## Scientific contract
The source decodes eight FP16 codebook weights per UInt16 code, uses three row-partitioned books with65536 entries each, and multiplies four input vectors. Genuine full12288×4096×4 source shape and96×512×4 signed dyadic probe are verified; no sampled-K surrogate. Synthetic source books/codes are not trained models. FP64 reconstructed quantized-matrix dots assess source arithmetic precision, not compression error versus unknown original dense weights or LLM quality.
Every inner pair has four fused half updates with RNEhalf intermediates, then FP32 lane accumulation and original warpdown16,8,4,2,1; finalhalfoutput. Ordinary Julia fma(Float16) maps llvm.fma.f16 in pinned backend. Paired scalarhalfFMA and unpackedhalfload mapping differ from source packedhfma2/int4 ISA; mathematical/rounding equivalence is verified, lowlevelISA equivalence is not claimed.

## Native original failure and correction
Originalhost allocates b*m/32=1536halfoutputs and offsets vectors bym/32=384, but kernelwritesm=12288pervector. A safeoversizedbacking diagnostic using unchanged kernel observed10752writes beyondoriginalTOTALlogicalcapacity, all16backingtailguards unchanged. This proves extent error without unsafeoriginalallocation execution or sanitizer claim; overlappingbatchstrides are source-proven.
Owned-copy minimal hostallocation/stride/debugcoverage repair defines corrected_reference_secondary. GPUkernelbodies/arithmetic unchanged, originalsource/failure retained. See native_quality_witness.json and native_corrected/output_extent_stride.patch.

## Numerical and scientific evidence
Final .porting-state/portable_julia/runs/20261018T093604113330Z_quantAQLM/result.json:20 full4,780,576UInt32 artifacts are bitidentical across allfourmodels andallfiveindependenttrials. Prefix4,730,912fields encodes everyactualsourceinput/code/book bit plusdeclaredcontrols (18,923,648exactbytes). Every49,536halfoutput matches independentlyexecuted CPUhalf-FMA/sourcewarptree oracle exactly;128tailguards/record. UInt32 is serialization of packedinputbitpairs and zeroextendedIEEEhalfoutputs, not mathematicalintegerarithmetic.
Sourcepositivebaseline uses originalFloat32division→half and glibcsrand123randcodes. Signedsecondprobe coversallthreepartitions, widecodeindices andhalfsubnormalrounding; onlydeclaredshapes certified. IndependentFP64matrixreconstruction/dot provides alternateprecision/reduction evidence. Alltwentyartifacts/source/fixturebindings rehashed; saved_output_audit.json and precision_quality_observation.json retainactual fullqualitymetrics.

## Timing, mapping and resource boundaries
SourceSMcountplanner retained:80SMV100 yields768blocks×512threads originalcase,48×64smallcase. Warp-per-row, padded32×9int4-equivalent sharedtile, uniformblockbarriers andsourcefloatwarp tree preserved. Native/CUDAjl/JACC request4608Bstaticworkspace plus4608Bunuseddynamicreservation. PinnedKAofficialcall has no shmemkeyword; KAusesrequired4608Bstatic only. This is an explicit resourcecovariate, not numericalunsupportedfeature; backend_API_review.json hashes actualsignature/mathoverride.
CUDAjl uses own framework shared/barrier/shuffle APIs, framework_device_kernel. KAusesownlocalmemory/barriers plusforeignCUDAwarp shuffle; JACC foreignCUDAshared/barrier/shuffle: both hybrid_device_intrinsics. One authoredkernel each, shared mathematical helper reused.
ROI sourcekernel launch+wholedevicecompletion, mean20 repeats pervector, sumfourvectors/twocases; source originally printedpervector time. JIT/state warmup outsidefivecheckedtrials, discardfirst/meanlastfour, allsamples retained. Transfers/CPUoracles/allocation outsideROI. No profiler/roofline/bottleneck diagnosis; packing/codegen/resource differences preventpurefrontendcausalclaims.

## Effort and reproduction
Reference restoration replaces historicalfalse-surrogate/unsupportedclaim; twohostprogramchecks, firstpreJITuploadAPIrejection and oneactualGPUcheck, oneauthoredGPUbody, zeroGPUcompilerfeedbackrepairs. Native two builds: guardedunchangedextentwitness andcorrectedprimary. Hostcodegenerator onebuild, CPUfixture/oracle generators separatelycosted. KA/JACC eachonecandidate/check andzero repairs after successfulreference/KA feedbackreuse; notglobalfirst-programsuccess.
Prospective reference/KA/JACC/workflow/annotation windows are context/cache/coordinationinclusive, historicalcostunknown. Savedbyte/sourceproxies notexclusivebilledtokens. Essential CPUfixtures andoracles are in ready_support_files.json withgenerator/hash metadata; actualGPUcaptures remainlocal.

Mean seconds: {"cuda": 0.0006480915375, "cuda_jl": 0.0016409315750000003, "ka": 0.0016394788374999997, "jacc": 0.0020023349000000005}

Experiment uses an owned study adapter invoking unchanged source GPUkernel and corrected hostgeometry; original CLI and its invalidallocation are not certified. Resourcebytecounts refer to declaredsource/APIreservations; actualgeneratedshared/register/occupancyattributes were notindependentlyprofiled.

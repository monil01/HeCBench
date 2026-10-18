# addBiasQKV: genuine FP8/BF16 fused attention preparation

## Operator and numerical evidence
The original default case is8192tokens,12heads,64features/head(hidden768), R1000. Each lane reads4FP8 E4M3FN bytes, converts to BF16, applies inputscale multiplication, BF16biasaddition, outputscale multiplication with BF16round-to-nearest-even after **each** operation, encodes finite-saturating FP8, and transposes [token,3,head,feature]→[token,head,3,feature]. A separate5token/3head/16feature/R10 axisprobe covers all254finite signed FP8 codes cyclically, non-BF16exact Float32scales(round to.5/1.25), asymmetricbias and axisgeometry. NaN/Inf/FNUZ/othergeometries and overflow-saturation execution are not certified.

Original inputs reproduce source mt19937seed123/uniform[-2,2] with original hostreference.h. A separate NumPy oracle enumerates finiteE4M3 values, finds nearest-even code independently, applies per-operation BF16bitrounding and independently reshapes the axes. It matches all18,874,368originalbytes and720axisbytes. Omitting the BF16roundings changes672,216original and33axis outputbytes. Therefore unrounded FP32 would violate the source contract. FinalFP8quantization versus roundedBF16value has maxabsoluteerror.125original/10axis, distinct from zero operator-byte error; no wholeattention/modelquality inference.

All40 accepted saved captures (2cases×4models×5) pass complete exact input prefixes, every outputbyte and384source/bias/outputguardbytes. Original fullartifact37,753,752bytes has18,879,000 exact input/parameterprefixbytes,18,874,368outputbytes and384guards; axis2136bytes has1032prefix/720output/384guards. Independentinput/oracleaudit and all8model/config sourcehash proofs are retained. Outputs are numerical FP8 storagebits, not printed PASS alone.

## Version-specific capability correction
Historical CUDA.jl only included unrelated mcpr; archived originals preserve S1 evidence. CUDA12.0sm70 rejected the BF16pair API and an sm80binary could not run V100. Those actual rejections did **not** prove a universal SM80 requirement. Root's CUDA12.9.41sm70 minimal+unchangedfullnative diagnostics passed on the same V100 and retracted that inference (`limitation_review.json`). Current canonical5trial evidence independently confirms the unchanged native kernel and all3Julia implementations on the original workload. No KA/JACC unsupported feature remains established.

All Julia versions retain UInt8FP8storage andUInt16BF16bias, compute oneFP32operation at a time with explicitBF16RNEbits before the next source operation, then perform FP8RNE encoding. This software arithmetic preserves the source lowprecision semantics; it is not silently unroundedFP32 replacement. Native BF16pair API itself does not establish a BF16hardware instruction. NativeCUDA12.9 vsJuliaCUDA12.0 and different source-level rounding representations are disclosed performance covariates; no assembly/resource profile/bottleneck claim.

## Mapping and measurement
Native uses oneblock/token with3D(D/4,head,3) threads; Julia versions use576flatthreads original/36axis and four logicalbytes/lane with equivalent axes. CUDA.jl uses hardwareblock/threadindices, KAGroup/Local indices, JACCglobalLinear division/remainder. All use0sharedbytes (JACCexplicitshmem_size=0), no ROI allocations. LogicalNTuple4 storage is preserved, without claiming actual emittedvector-instruction widths. CUDAhostallocation/copy/completion is glue; devicebody uses normal Juliafloat/bitoperations, no foreignCUDAdeviceABI/vendorattentionlibrary.

Each case warms a wholeRlaunch loop, then5Rlaunch loops with devicewideblockingcompletion; mean cases2–5 is divided byR. Hostlaunch/completion overhead is included, buffers/guards/CPUoracle/readback/serialization are outsideROI. All4 use the same fullydefined sharedinputs and original unchanged kernelbody. The original default operator is certified, not every genericCLI/exceptionalbitpattern.

### Mean per launch, microseconds

|Case|CUDA|CUDA.jl|KA|JACC|
|---|---:|---:|---:|---:|
|original|289.209|193.586|194.753|212.033|
|axis|5.361|6.674|9.489|15.229|

## Effort and retained evidence
Fresh CUDA.jl restoration:1candidate/1generic kernel/2actualGPUJITcheckinvocations,0compiler/runtime/numericalfeedbackrepairs; replacing the historicalalias is1semanticrestoration, historicaleffortunknown. Native2currentbuilds and CPUfixturegenerator1build pass; older12.0failures/root12.9review are separate sharedhistory. KA thenJACC each1candidate/1generic kernel/2actualGPUJITchecks/0repairs. Shapes, five trials and internalR1000 are not compilerrepairrounds; internalcompilerpasses/specializations unknown.

Prospective phase usages include context/cache and coordination; sharedworkflow overlapschildren and is nonadditive/notexclusivebilledcost. Referencephase includes a seam-carving provenance-metadata followup; preserved in meter notes. Partial saved-source proxies remain separate. Previous blocked manifests are archived under pertarget prior_cuda120_unattempted; oldfailureevents and limitation_review remain unchanged, with explicit resolution events. See fixture_provenance.json, audit_fixture.py, saved_output_audit.json, source_integrity_review.json and percase configs. No measured outputcaptures are added to source Git; only reproducible input/CPUoracle fixtures/generators/hashes.

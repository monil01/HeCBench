# seam-carving: full source image, three100-seam modes

## Scientific and numerical contract
Original STB-decoded `Broadway_tower.jpg` is a128×128 BMP despite its suffix. All4 implementations remove100vertical seams to28columns at fixed128pixel pitch, in standard, incremental-cost-update, and approximate-greedy modes. RGBL1 forward costs use Int16storage/Int32DP and integer indices; packed UInt32RGB bytes preserve the original uchar4 layout, with unused alpha canonical0. Historical CUDA.jl mcpr alias is archived and replaced by nine genuine device kernels, not a framework limitation.

All60 saved arrays match the source-behavior CPU emulator exactly, all5 comparisons against both CUDA and CUDA.jl pass with exact16,391-word input prefixes. Captures contain the finalRGBimage/lastseam plus defined finalDP/costs or greedy maps:34,951/34,567/31,152 UInt32values for standard/update/approx. Independent checks prove every final row is an ordered subsequence of the input row and the last seam is valid/adjacent. This is not a claim that every target saved all100intermediate paths.

## Source quality is separate from translation correctness
A separate untimed actual-native GPU diagnostic saved all100paths and actualpreseamRGBimages per mode. Independently evaluating predecessor-aware forward energy gives95valid but nonminimum paths in both standard/update, maximumenergygap132. First bad step1(width127) has observed1292 versus validoptimum1276. Sourcebacktracking minimizes previousM without adding currenttransitioncost; its logicalright condition is also offbyone, but no invalidpath or allocatedOOB was observed on this fixture. These unchanged-native source-behavior configurations are explicitly excluded from ideal minimum-energy scientific-fidelity aggregates. Approximate mode has69objectivegaps/max20054: expected greedy approximation P1, not a failed translation. No human perceptual quality/PSNR/SSIM claim.

Unused approximate halo cost entries are given defined values in Julia; only interior lanes1..30 consume costs. A dead lastrow index read is elided; no claim about unspecified native alpha/unwritten cells. Otherimages/shapes or width>256tiledDP are unported/unmeasured, so originalgenericCLI is not certified.

## Measurement and mapping
One full100seam warm case precedes5fresh-input cases; means use cases2–5. Inputreset/allocations/H2D are outside ROI. Full100seam dispatchloop, D2D index resets, pixel/cost pointer swaps, and finalincrementalcostupdate are inside. Updateinitialcost is fully completed outside ROI; originalCLI lacks that pre-ROI synchronization, so these matchedloop-only measurements are not its coldCLI timer. All Julia boundaries use devicewideblockingcompletion. Native kernels/wrappers are unchanged.

CUDA.jl preserves original2D(32,8) tiles. KA/JACC flatten256 lanes while preserving warp-x/row mapping. DP uses actualworkgroupwidth128→29 in allmodels; native/CUDA.jl/JACC sharedcapacity is8*cw bytes, KA uses fixedmaximum1024B with supported runtimeworkgroupsize. Cost/minreduce/approx sharedarenas are3072/1024/4608B. JACC explicitly allocates exacttemplate capacities and perwidthLaunchSpecs outside ROI; shared(template) adds copy/barrier work relative to the other implementations. Zero shared bytes are explicit for remove/update/jump/backtracks. No unsupported capability is inferred from this representation choice. No profiling evidence supports a bottleneck/roofline attribution.

### Mean whole100-seam case time, milliseconds

|Mode|CUDA|CUDA.jl|KA|JACC|
|---|---:|---:|---:|---:|
|standard|8.564|8.513|8.990|9.432|
|update|8.639|8.605|9.024|9.079|
|approx|5.456|5.800|6.195|10.128|

## Effort and provenance
CUDA.jl restoration:1candidate,3actualGPU/JIT check invocations,9generic kernel definitions, noGPUfeedbackrepairs; historicaleffortunknown. Native4builds including1diagnosticbuild,3untimedmode diagnostics. OriginalSTB hostdecoder had1API compile rejection then repaired to source`free`; separate fromGPUtranslation/JIT. KA and JACC each1candidate,3actualGPU/JIT check invocations,9genericdefinitions,0repairs;100widths are not100repairrounds, internalcompiler specializations unknown. Shared helpers/reference/CPU quality audit counted separately; JACC reuses successful KA mapping.

Usage windows include context/cached tokens and coordination, overlap broad workflow, and are not additive/exclusivebilled costs. Separate output source proxies are incomplete final-artifact estimates. Raw runs, failedCPUdecoder source/log, source hashes, generators and native quality witnesses are retained. See `saved_output_audit.json`, `source_integrity_review.json`, `fixture_provenance.json`, `diagnostics/native_quality_witness.json`, configs and phase manifests.

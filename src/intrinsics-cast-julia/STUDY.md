# Intrinsics-cast study

## Scientific and numerical contract

This is a GPU type-conversion/rounding microbenchmark, with no identified scientific application or dataset. Both original device regions are restored:37 from-floating and31 to-floating intrinsic call sites, including four directed rounding modes, signed/unsigned32/64-bit conversions, high/low word reconstruction and bitcasts. The output is an Int64 aggregate, not an ordinary numerical estimate. Compiler simplification may remove/reuse operations; sourcecall counts are not measuredinstructioncounts/FLOPs.

Original native host initialization loops1..n, leaving input[0] undefined and writing input[n] beyond its logicalallocation. A literal-loop safe n=1 guard-slot witness records the defect; it does not claim an address-sanitized originalexecution. The original CLI is defective and excluded. A new fully defined shared fixture loads65536 sourceformula22.44/(i+1) doubles or65536 copies of original0x403670A3D70A3D71 integerword, with unchanged native devicekernel bodies. This namedfixture is not certification/correction ofthe originalwholeprogram. Defaultmillion-element/scaling and NaN/Inf/negative/out-of-rangefloat inputs/otherintegerwords remainunmeasured. ThetoFPfullarrayhasoneuniqueinputword; repeatedelementsarenot independentconformancecases.

Python CPUoracle implements exactbit operations/modularC++promotions, binary32storage via struct and directedinteger->float conversion by nearestconversion thenexactintegercomparison/neighborstep. Every65536Int64outputs, eachof5trials, bothregions, matchCPUbytes andbothGPUreferences exactly. This is aggregate-program validation, not exhaustiveindividualintrinsicconformance. NativeCPUcompare usesexactInt64 equality. SharedOwnedStudy scalarCPUmetric coercesInt64toFloat64, which canmasklowbit errors; separate audit_cpu_bits.py checks allsavedbytes against independentexpected andretains artifacthashes. Noextratimingserieswasneeded.

## Implementations and timing

Archived Julia omittedmostconversions andprinted hardcodedCUDAchecksums/unconditionalPASS. Currentreference restores fullsemantics, C++narrowing/wrap andFloat32/64accumulationorder. KA/JACC reuse these sharedreference helpers: explicit CUDA libdeviceABI calls for directedrounding, so allJulia-family versions are hybrid_device_intrinsics, notbackend-independent genericmath. Bitcasts/exact32bitint->double useJulia primitives. No attemptedunsupportedgenericAPI orframeworkfailureisfabricated.

All4 use256blocks256threads, globalinput/output, no sharedmemory (JACCexplicit0),100overwrite launches/outertrial, explicitJITwarmup,5inprocesssynchronized host-clocktrials andmeanlast4. Inputs/allocations/resets/transfers/checks/artifacts areoutsideclock. Pair1MiBworkingstoragefitsV100L2; noDRAM/roofline/occupancycauseclaimwithoutcounters. Bothsourcekernelregionsmeasuredseparately.

KA/JACC eachcandidate1/2actualchecks/2kernels/0repairs. Sharedcurrentreference is1semanticrevision/2checks/2nativebuilds plus1hostwitnessbuild, withallconversionhelper/Pythonoracle generationchargedthere. Historicalgenerationeffortunknown. Cleanmeterordering closesKA beforeJACCfirstdraft; JACCreusesKAfeedback. Partialsourceproxiesremainseparatefromcontext/cache-inclusiveconsumedusage; workflowoverlapschildwindowsandisnotadditiveormonetarybilling.

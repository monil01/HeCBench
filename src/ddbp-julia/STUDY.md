# Distance-driven tomographic backprojection: original failure and corrected secondary

## Source-backed imaging operator

Rodrigo de Barros Vimieiro's GPLv3-or-later2019source describes halfcone-beam DBT distance-driven backprojection, citing branchless projection/backprojection work. The actual eight GPU stages pad projections, map detector/objectboundaries, integrate2Dprojections, rotate detectorcoordinates, map detector rays to eachslice, bilinearlysample integralprojections, mixedfinite-difference withcosineangle/voxel-detectorsize weighting, thennormalizeoverviews. This is an imaging numericaloperator, not proof of clinical/medical imagequality.

## Actual unchanged-native counterexample

One explicitly diagnostic unchanged-native8×8×1/oneview/unitprojection/zeroangle/z0 unitspacing run produces55wrongvoxels of64 versus analytic1/(cosatanx*cosatany), maxabs4.3857725215, finitebutnegative/zerooutputs. `native_quality_witness.{cu,json}` and `native_quality_witness_logs.json` retain exact source/harness/log/rawbinaryhashes, oneillustrativescalar andindependentanalyticvalues. ActualfullGPUbinary remainslocal, notsourceGit. OriginallargeCLI wasnotexecuted, no invented failure on its huge default. Static source shows in-placeinter-threadscanwithoutbarriers and offset9stepswith4workerextent skippingrows/columns; actualracecausality notisolated. Nativechecksumprinting alone didnotprovide mathematicalvalidation. HistoricalJulia is a different sin²volumegenerator ignoringprojectiondata, archivedunchanged.

## Explicit numerical-method intervention

Originalprimary remains **reference_blocked**, notclaimedfour-wayagreement. `native_scan_correction.patch` changes onlyscanbody andtwoaxislaunchsections: one GPUworker owns anentireline, serialinclusiveprefix; fullcoverage, twoorderedaxislaunches. Allother7kernels unchanged. Originalsource neveredited. The acceptedresult is **corrected_reference_secondary**, native_reference_repaired=true with canonicalreference_provenance original/corrected/patch/actualwitnesshashes. Do not fold into untouched-nativeperformance or describe correction as frameworkunsupported. Originalpadding usesnPixY ratherthanDetY; bothdeclaredcases useequal8soallinputpaddingdefined, nPixY<nDetYuncertified.

## Complete independent stage verification

Twofixtures: analytic8×8×1/1view unitplane, and signed-angle threeview7×8×2 volume/8×8detector, nonuniformpositive dyadicprojectiondata. Independent NumPy2.5.4 oracle uses globalaxis cumsums, parametricray–plane intersection, tensorproductbilinearweights andsqrtanglefactors (rather thantranscribingstageCUDAcode); unitplanehasclosedformcontrol. All2867intermediatevalues overboundaries, bothintegrations, eachviewrotation, eachslice mappedfields/interpolatedintegral/cumulativevolume andfinalnormalizedvolume checked atol1e−10+rtol1e−9. Actualmaxindependenterror≤4.085621e−14 across20savedcaptures. Complete360input/parameterprefixvalues and832guardwords exactbytes independentlyaudited; everymeasuredsource/fixturebinding rechecked. Stagecaptures are anuntimed matchingpipeline rerunoutsideROI, nothiddenDTOHtimedwork. EssentialCPU-generated study_proj{1,2}_f64.bin/oracle_case{1,2}_f64.bin havegenerator/hashmetadata; GPUarchiveslocalonly. Notoriginal1996×2457×78/15viewsizeorclinicalqualitycertification.

## Abstractions and measured region

All3Julia models: mixed_host_device_pipeline withframework_device_kernel component. Onegenericfrontendkernel has8Valstagehelpers/specializations, ordinaryCUDA/KA/JACC3Dindices, same256×1×1/16×16×1/8×8×4threadgroups, noatomics/sharedmemory. Actualsource45stagecalls across2fixtures perreconstructionpair, correctedscanexplicit. JACCexplicitshmem_size0/syncfalse; its3Dpath defaulttaskstream used, devicewidecompletion closesROI (no customstreamclaim).

Originalpreallocated timingregion retained: padding/H2D/memsets/maps/scans/hostviewtrigonometry/geometry/interpolation/differentiation/normalize; allocation/free/finalDTOH outside. Hence performanceincludes host/copy/control work, notjustGPUkernel latency orpurefrontendoverhead. NativeInt32vsJuliaInt64indices/hostbindings/containers/zerofill implementation remaincovariates, no bottleneck/rooflinecausation asserted. Fivecheckedinprocessoutertrials, fullwarmup,3complete reconstructions eachcase/outertrial, meantrials2–5. Headline sumtwo componentmeans, allsamples retained. Actualseconds: {"cuda": 0.00039199766666675, "cuda_jl": 0.000327373, "ka": 0.0006263358333332501, "jacc": 0.0011272250833325001}.

## Attempts and token activity

Onecurrentreferencecandidate/processJITevaluation, KAone, JACCone, allfirstcandidate pass withnocompiler/runtime/numericalfeedbackrepair. Eightstagecases are not eighttranslationattempts; internalcompilerjob/passcountunknown. Historicalaliasrestoration separatefromnativeobservedfailure/onecorrectedscanintervention. Oneunchangednativewitnessbuild/run separatefromoneacceptedcorrectednativebuild. Prospectivecontext/cache-inclusivephasewindows andoverlappingbroadworkflow separatefromexclusivebilledcost and savedsourceproxies; historicaltokens unknown.

Finalbundle `.porting-state/portable_julia/runs/20261018T052043594655Z_ddbp/result.json`, configstudy_config.json withrole/provenance, saved_output_audit.json andsource_integrity_review.json. PinnedJulia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,V100sm70;make runserializesGPU6. Coordinator owns separatecommits/manuscriptupdates.

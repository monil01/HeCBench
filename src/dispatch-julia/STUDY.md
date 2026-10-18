# Empty-kernel dispatch: execution/timing validation, no numerical outputs

## Source contract

Native `EmptyKernel()` is genuinely empty, withnoarguments/arrays/numericinputs/outputs; historicalCUDAjl alreadymatches. All3Julia versions retainactualnormalGPUdispatch, oneblock/onethread. No fake checksum, sentinelallocation orCPUoracle is introduced. `numerical_agreement_kind=no_numerical_outputs_execution_only`, status **execution_only_verified**, checkednumericvalues0, nullerrors, numericalaccuracyinapplicable. Keepseparatefromnumericandproperty-equivalent counts. Sourceemptybody/hashproofplusactualsuccessfuldriverlaunch/fullcompletion establishexecutionevidence, not scientificcomputationcorrectness. No targetcompiler/runtime/numericalfailuresoccurred. One postverificationlocalmetadatahelper TypeError(maxofnullerrors) was fixedinprivatehelpercopies, preservesnullmetrics, noGPUrerun/candidatechange; separatelyrecordedT1event.

## All original timing modes

Asynchronoushostenqueue: timerwrapsonlylaunch, phasecompletionoutsideROI. Single-dispatchlatency: launch+wholedevicecompletioninsideeachsample. Batchdispatch:1000launches+wholedevicecompletion, reportedperlaunch. Original100warmups+1000retainedsamples per mode,1100fullsamplesretained; sourcecountsunchanged. Fivecheckedinprocessoutertrials, firstoutertimingexcluded, meanlast4. Initialfullcompile/launchwarmupoutsideallROI, sourceinternalwarmupseparatefromouterprotocol. Headline isbatch-per-launch, notsumofunlikeenqueue/singlelatencies. Sourcecompletedloops1,102,200hostlaunchcalls/outertrial,5,511,000permodel; notindependentprofilerGPUblockcounter.

All66,000rawtimes(20records×3modes×1100samples) retainedinresultJSON/logs. Independentread-onlyaudit recomputesallmeans/populationstddevs andlast4summary, checksdeclaredworkcounts, actualsuccessfulAPIchecks,16source/proofbindings. Native originalFloat32summarystatistics normalizedtoFloat64likeJulia; no devicearithmeticchanged, sourceprintsnotclaimedbyteidentical. No outlier/fastestselection.

## Per-model mapping and interpretation

CUDAjl normal@cuda; KAcachedemptykernelobject/ndrange1; JACCparallel_for1withprebuiltLaunchSpecthreads1blocks1shmem_size0syncfalse. Allframework_device_kernel, noforeigndeviceintrinsicsorvendorcompute. CPUtimers/statsFloat64, GPUcompute/storageprecisionabsent. Emptyworkmakesroofline/FLOPs/traffic/AIinapplicable, nobandwidthorapplicationaccuracyclaims. DispatchhostAPI/cache/resourcequery/driver/completioncosts areinregion; noprofiler-basedcausalbreakdownclaimed. Fine-grainruntimegranularityusecaseonly.

Actualmeansecondsperlaunch components(meanouter2–5):

{
  "cuda": {
    "enqueue": 1.770991e-06,
    "single_dispatch": 7.3281105e-06,
    "batch1000_per_launch": 1.9602887985e-06
  },
  "cuda_jl": {
    "enqueue": 2.24343025e-06,
    "single_dispatch": 7.97735675e-06,
    "batch1000_per_launch": 2.24970294775e-06
  },
  "ka": {
    "enqueue": 3.01151375e-06,
    "single_dispatch": 8.9807675e-06,
    "batch1000_per_launch": 2.79854674225e-06
  },
  "jacc": {
    "enqueue": 4.93234625e-06,
    "single_dispatch": 1.126049525e-05,
    "batch1000_per_launch": 5.00141218675e-06
  }
}

## Agent effort and reproducibility

Reference/KA/JACC eachonegenericdevicecandidate/processJITevaluation, no compiler/runtime/numericalfeedbackrepair; historicalkernelalreadyfaithful, onlymeasurementadapter/sourceproof/rawtimingadded. Historicalattempts/tokensunknown. Prospectivecontext/cache-inclusiveusagephasesseparatefromnonexclusivebroadworkflow/sourceproxies, notexclusivebilledtranslationrankings.

Finalbundle `.porting-state/portable_julia/runs/20261018T053228734981Z_dispatch/result.json`, study_config.json/empty_kernel_source_review.json/model-specificproofs/execution_timing_audit.json/source_integrity_review.json. NoGPUoutputartifacts existorexpected. PinnedJulia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,V100sm70. `make run` serializesGPU6. Coordinatorownsseparatecommits/papercensusintegration.

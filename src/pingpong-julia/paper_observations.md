# Pingpong research observations

## RQ1

{
  "contract": "BothMPI/NCCL,twoactualGPUs5+6,fullbothrankactualinputs/after5increments/after50roundtrips, completeguards; all20primarystrictexactdualsandindependentNumPypassed.",
  "stage_values_all20": 330363120,
  "raw_guard_bytes_all20": 491520,
  "scope": "Both MPI/NCCL, two local ranks onphysical5+6, bounded65536/262144/1048576/257 Float64 sizes. Original512KiB..1GiB fullsweep unmeasured; not inter-node MPI scalability.",
  "no_scientific_simulation_claim": true
}

## RQ2

{
  "CUDAjl": "CPUfrontend2processes/1B2syntaxreject; sharedconfigreuseT1beforebuild;1complete2rankGPUprogram pass/1authoredkernelbodybundle(2defs).",
  "KA": "1complete2rankprogram/sourcebundle pass,2ownframeworkdefs.",
  "JACC": "5complete2rankprogramversions,4authoredmainversions,1unchangedGPUbodybundle/2defs;4authoredJITprocesses acrosslast3groups; noactualGPUcompilerreject. SharedPIDstartup+targetarrayAPI+targetstreamlifecycle+boundedNCCLwait preserved; eagerbeforecommsetup restoresobservedsuccess.",
  "actualtokens": {
    "cuda_jl": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 2339475,
      "output_tokens": 5566,
      "cached_input_tokens": 2326144,
      "reasoning_output_tokens": 1920,
      "total_tokens": 2345041,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 128047414,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/pingpong/cuda_jl_translation.jsonl",
      "marker_lag_seconds": 6.138822,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "ka": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 1746068,
      "output_tokens": 1840,
      "cached_input_tokens": 1740032,
      "reasoning_output_tokens": 415,
      "total_tokens": 1747908,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 128125062,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/pingpong/ka_translation.jsonl",
      "marker_lag_seconds": 8.720164,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "jacc": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 7965245,
      "output_tokens": 16631,
      "cached_input_tokens": 7919104,
      "reasoning_output_tokens": 8372,
      "total_tokens": 7981876,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 128667006,
        "lookup_bytes_reviewed": 4194304,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/pingpong/jacc_translation.jsonl",
      "marker_lag_seconds": 15.519274,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    }
  },
  "limits": "Contextinclusive actualdeclaredwindows; cached/reasoning subsets,nestedverificationoverlap; JACCmixedsharedstartuprepair, allmodels2GPUintegration excludedcleancalibration. Nohistoricalactualreconstruction."
}

## RQ3

{
  "mean_seconds_last4": {
    "cuda": 0.0018803798474999955,
    "cuda_jl": 0.0018710528300000108,
    "ka": 0.0018868095075000046,
    "jacc": 0.0018705262000000022
  },
  "metric": "Sum8rank0per-transfermean timings; notisolated1Mbandwidth orpureGPUkerneltiming",
  "individualcase_data": "src/pingpong-julia/per_case_communication_times.json",
  "limits": "Fixedorder,localhost,twoGPUvendorcommunicationstratum separatedfromsingleGPU/purekernelaggregate; nointer-node scalabilityclaim."
}

## RQ4

{
  "observations": [
    "GlobalLaunchSpec capturespre-device stream; rank1invalidresourcehandle fixedbycurrentdevice streambinding.",
    "ExplicitrankPIDbootstrapbeforeglobalCUDAcontextcreationkeepsstrict externalguard.",
    "JACC.Array isBase.Array; actualbackendallocationAPI lowercaseJACC.array; CPUfallbackneveraccepted.",
    "Eager2kernelsetupbeforecomm restoredall5 afteractualfirstNCCLwarmuphang; exactmechanismnotisolated."
  ],
  "not_established": [
    "intrinsicframeworkcommunicationpenalty",
    "causalNCCLJITdeadlockmechanism",
    "actualinterconnecttransportselectionorroofline"
  ]
}

## RQ5

{
  "selection": "All3can supplyownallocation/incrementkernels andJuliaMPI/NCCLhostloops, but vendorcommunication controls primaryROI. Correct perdevice streams, cold-JIT placement and CUDA-awarehandoff are operationalrequirements.",
  "backend_dependency": "MPI/NCCL/CUDAhostinteroperability explicit, no crossvendor claim."
}

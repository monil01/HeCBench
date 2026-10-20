# MMCSF qualified precision-secondary observations

## RQ1

{
  "original_max_abs_error": 0.04012859,
  "unchanged_strict_budget": ".01+2e-5*abs(exactoracle)",
  "intervention": "OnlyoutputatomicF64, allprivatecontributionsF32; same-modeallpartitionaccumulation plus includedprojection",
  "max_F64_error": 0.00038338403010129696,
  "max_projected_F32_error": 0.00038695353805451305,
  "full_captures": 20,
  "values_per_capture": 4514748,
  "exact_integer_CPU_replay": "pass allrows/modes/columns; everyCPUgold0difference",
  "native_bug_proven": false
}

## RQ2

{
  "candidate_units": {
    "cuda_jl": {
      "whole_program_evaluations": 2,
      "actual_device_JIT_processes": 1,
      "mathematical_bundles": 1,
      "kernel_entries": 7,
      "source_generator_invocations": 2,
      "source_generator_failures": 1,
      "host_adapter_failures_before_deviceJIT": 1
    },
    "ka": {
      "whole_program_evaluations": 1,
      "actual_device_JIT_processes": 1,
      "mathematical_bundles": 1,
      "kernel_entries": 7
    },
    "jacc": {
      "whole_program_evaluations": 1,
      "actual_device_JIT_processes": 1,
      "mathematical_bundles": 1,
      "kernel_entries": 7
    },
    "native_secondary": {
      "build_candidates": 1,
      "readiness_executions": 1,
      "five_trial_processes": 1
    },
    "backend_internal_compiler_invocations": "unknown; not inferred from processcounts",
    "ordered_reuse": "SevenCUDAjlmathhelperbodies reusedbyKA/JACC, originalnativeCPUcontroller/review reused; not21independenttranslations."
  },
  "observed_boundaries": {
    "cuda_jl_translation_feedback": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/cuda_jl_translation_feedback.jsonl",
      "begin_event_id": "bb76d7f6874d6d701635eef5",
      "end_event_id": "f5522ceb997e456577577406",
      "observed_usage": {
        "input_tokens": 2018684,
        "cached_input_tokens": 1995264,
        "cache_write_input_tokens": 0,
        "output_tokens": 9422,
        "reasoning_output_tokens": 3293,
        "total_tokens": 2028106
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:31:31.861Z",
        "context_event_sha256": "58a5a65cc9434441b189991452f9f4ed157790709e136ffdb77f3f659292f2b2",
        "source_snapshot_bytes": 135977929,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": "CUDAjl all7 native-equivalent mixedprecision secondary kernels pass all5fullscope scientific/fullarray/nativecomparisons; first hostnulloptionalview error beforedeviceJIT preserved, oneunchanged mathematicalbundle. Shared AST math generator authored here and reusedlater; originalprimaryblocked. Root-onlycommitpending."
    },
    "jacc_translation_feedback": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/jacc_translation_feedback.jsonl",
      "begin_event_id": "2ddba09fe0467e393048aae7",
      "end_event_id": "2dc168e7c8f4aa5c246c313d",
      "observed_usage": {
        "input_tokens": 1255782,
        "cached_input_tokens": 1245056,
        "cache_write_input_tokens": 0,
        "output_tokens": 4141,
        "reasoning_output_tokens": 2419,
        "total_tokens": 1259923
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:31:31.861Z",
        "context_event_sha256": "58a5a65cc9434441b189991452f9f4ed157790709e136ffdb77f3f659292f2b2",
        "source_snapshot_bytes": 136238266,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": "JACCfirstcandidate all7/all4case/allmode fullsourcepipeline passes all5fulloutputs/actualinputbits/guards/strictindependentoracle and exactdualrefs; final084209583131Z matched_performance_validtrue. Explicit0dynamicshared and per(op,grid,block)LaunchSpec, CUDAatomic/indexbackendextension. No otherbenchmarkwork/sharedharnesscreation; inheritedmath/hostreuse disclosed."
    },
    "ka_translation_feedback": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/ka_translation_feedback.jsonl",
      "begin_event_id": "781320178155fdf28068d209",
      "end_event_id": "730690b3983e4ac03bc04395",
      "observed_usage": {
        "input_tokens": 1159970,
        "cached_input_tokens": 1148800,
        "cache_write_input_tokens": 0,
        "output_tokens": 3420,
        "reasoning_output_tokens": 1674,
        "total_tokens": 1163390
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:31:31.861Z",
        "context_event_sha256": "58a5a65cc9434441b189991452f9f4ed157790709e136ffdb77f3f659292f2b2",
        "source_snapshot_bytes": 136110164,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": "KAfirstcandidate all7ownlogicalindexed kernels passes all5fullscope independentgates and exactfullarray vsboth native/CUDAjl; unchanged inherited mathematicalbundle, CUDAatomic backendextension explicit. No otherbenchmarkwork/sharedharnesscreation insidewindow; orderedCUDAjlmath/controllerreuse."
    },
    "shared_independent_accuracy_review": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/shared_independent_accuracy_review.jsonl",
      "begin_event_id": "75ff0fe6dd8c001314cf4a7c",
      "end_event_id": "57eeaa3fe304c29cd9636657",
      "observed_usage": {
        "input_tokens": 2846558,
        "cached_input_tokens": 2803456,
        "cache_write_input_tokens": 0,
        "output_tokens": 12724,
        "reasoning_output_tokens": 7314,
        "total_tokens": 2859282
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:27:44.580Z",
        "context_event_sha256": "255d62358cb6473423a661cca67e017816c42d97928b73c699e6e3143af17a92",
        "source_snapshot_bytes": 131495085,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": ""
    },
    "shared_native_readiness": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/shared_native_readiness.jsonl",
      "begin_event_id": "132f79f79ee30adea67f66f1",
      "end_event_id": "3656ec39b28a2f88d82fde36",
      "observed_usage": {
        "input_tokens": 7255139,
        "cached_input_tokens": 7146496,
        "cache_write_input_tokens": 0,
        "output_tokens": 61284,
        "reasoning_output_tokens": 35657,
        "total_tokens": 7316423
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T03:11:18.213Z",
        "context_event_sha256": "f5f484d6bd68adaa813d62acfb2221bfb1562a5d4484835bdf86fffc91a5da1a",
        "source_snapshot_bytes": 116199992,
        "lookup_bytes_reviewed": 4194304,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": ""
    },
    "shared_reference_resolution": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/shared_reference_resolution.jsonl",
      "begin_event_id": "3acc00027e2609a1b98074ea",
      "end_event_id": "2a8fef82ba0b7deafc4f3582",
      "observed_usage": {
        "input_tokens": 3176884,
        "cached_input_tokens": 2789376,
        "cache_write_input_tokens": 0,
        "output_tokens": 15558,
        "reasoning_output_tokens": 8392,
        "total_tokens": 3192442
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:31:31.861Z",
        "context_event_sha256": "58a5a65cc9434441b189991452f9f4ed157790709e136ffdb77f3f659292f2b2",
        "source_snapshot_bytes": 135699967,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": "Qualified FP64 output-atomic secondary native full readiness passed all7branches/all4cases/allmodes strict unchanged budget; original primary blocked. New native five-trial series submitted, not yet promoted. Shared harness/config creation/native math review separate from targets."
    },
    "shared_verification": {
      "boundary_file": ".porting-state/portable_julia/usage_boundaries_300/mmcsf/shared_verification.jsonl",
      "begin_event_id": "192d18c99e29137cb2927ff0",
      "end_event_id": "c1283a92b498574153d4eec8",
      "observed_usage": {
        "input_tokens": 396897,
        "cached_input_tokens": 392320,
        "cache_write_input_tokens": 0,
        "output_tokens": 4652,
        "reasoning_output_tokens": 2187,
        "total_tokens": 401549
      },
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T08:31:31.861Z",
        "context_event_sha256": "58a5a65cc9434441b189991452f9f4ed157790709e136ffdb77f3f659292f2b2",
        "source_snapshot_bytes": 136323942,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "scope": "Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.",
      "note": "CPUexactsignedintegerFloat32COOoracle all20full4.514Mvaluecaptures/all37uniqueexecutedsourcepaths checked; every storedlongdoublegold Float64 matchesexactintegeroracle0difference; maxFP64.000383384/projectedFP32.000386954, unchangedstrictbudget. NoadditionalGPU/rerun; allmodelsmeanslast4 checked."
    }
  },
  "interpretation": "Cachedinclusiveorderedworkflowtokens, notindependentframeworkcost; CUDAjlnewsharedhelper excludescalibration."
}

## RQ3

{
  "mean_seconds": {
    "cuda": 6.507031856,
    "cuda_jl": 8.084757395,
    "ka": 6.113310944,
    "jacc": 7.43556578525
  },
  "raw_five_seconds": {
    "cuda": [
      6.648155877000001,
      6.56205523,
      6.330828064,
      6.342608289,
      6.792635840999999
    ],
    "cuda_jl": [
      7.296597722,
      7.034831251,
      7.087939983999999,
      8.508148517999999,
      9.708109827000001
    ],
    "ka": [
      6.175130934,
      6.123747676000001,
      6.165308445,
      6.136662385,
      6.02752527
    ],
    "jacc": [
      9.830580821,
      8.739526355999999,
      6.8136961000000005,
      7.045096081,
      7.143944604
    ]
  },
  "timing_scope": "qualified_full_source_pipeline_including_FP64_output_allocation_projection_minus_added_observations",
  "exclusions": "NotuntouchedFloat32native/no device-only orfullapplicationperformanceclaim; preserveallspikes."
}

## RQ4

{
  "motif": "Irregularsparsefibercontraction/scatteratomicreduction; precisionsecondarysupportedbyallroworacle andboundedindependentrow762CPUroundoffcontrol, notGPUatomicordertrace.",
  "resource": "128threads/block,7sourcekernels,0shared; JACCLaunchSpeckeyedperop/grid/block",
  "causal_limits": "No uniqueerrorcauseproof/no measuredroofline/bottleneck attribution."
}

## RQ5

Frameworkselectiondoesnotbyitselfrepairaccuracy; fullindependentcontractdetectedsharednativeFloat32error. All3modelsrequireCUDAatomicbackendextensioninthismapping andsamequalifiedprecisionpolicy. Broaderportabilityunprobed.

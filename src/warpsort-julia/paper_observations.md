# Warpsort research observations

## RQ1

{
  "result": "All20 complete258-caseblocks accepted with exactGPUdualrefs; numericaux source signedzeros equivalence, full rawbitoutputs retained.",
  "cases": "Everylength1..128 bothvariants; source14special\u00b1Inf/\u00b10/subnormal controls, duplicatekeys37/74/111; noNaNs.",
  "outputs_pertrial": 24810,
  "input_prefix_values_pertrial": 17056,
  "guard_bytes_all20": 3962880,
  "qualification": "Post-observation overstrong auxiliary signedzero bitmultiset rejection preserved; source GPUalgorithm unchanged, native_reference_repairedfalse. No >128or original1000shuffle CLI claim."
}

## RQ2

{
  "CUDAjl": "4completeprogram candidates,2authoreddevicebody candidates/1mathalgorithm,2authoredsortJITprocesses (1InvalidIRreject); first2programs hostsyntax/copyrejects beforeauthoredJIT. CPUparse-onlyfalsepass preserved.",
  "KA": "1completeprogram/1bodycandidate/1authoredJITprocess pass",
  "JACC": "1completeprogram/1bodycandidate/1authoredJITprocess pass",
  "actualtokens": {
    "cuda_jl": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 4385300,
      "output_tokens": 10516,
      "cached_input_tokens": 4356864,
      "reasoning_output_tokens": 2304,
      "total_tokens": 4395816,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 127392226,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/warpsort/cuda_jl_translation.jsonl",
      "marker_lag_seconds": 12.995481,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "ka": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 1180704,
      "output_tokens": 2066,
      "cached_input_tokens": 1175040,
      "reasoning_output_tokens": 86,
      "total_tokens": 1182770,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 127461763,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/warpsort/ka_translation.jsonl",
      "marker_lag_seconds": 6.006094,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "jacc": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 2176521,
      "output_tokens": 2978,
      "cached_input_tokens": 2168320,
      "reasoning_output_tokens": 1070,
      "total_tokens": 2179499,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 127581813,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/warpsort/jacc_translation.jsonl",
      "marker_lag_seconds": 6.150712,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    }
  },
  "limits": "Actual context-inclusive activity windows include sharedGPUlockwait/coordination/polling; cached/reasoning subsets, nested verification overlaps, backendcompilerinvocationsunknown."
}

## RQ3

{
  "mean_seconds_last4": {
    "cuda": 0.0051804265000000025,
    "cuda_jl": 0.0066306715,
    "ka": 0.0074916672499999995,
    "jacc": 0.009159557
  },
  "scope": "Sum258 source launch+wholecontext completion intervals perblock, setup/alloc/reset/copies/fulloracles outside; fullwarmup then5checked blocks mean2\u20135",
  "limits": "Aggregate258tiny launches, no singlekernel bandwidth/throughput or causalframework overhead claim."
}

## RQ4

{
  "observed_mechanism": "Immutabletupleinitializer fixes boxedloop-mutatedclosure InvalidIR, scientific algorithm unchanged. CUDAjl ownexportedshuffles are framework features; KA/JACC depend on CUDA.jl intrinsics explicitly hybrid.",
  "not_established": [
    "causal performance attribution",
    "crossvendor portability",
    "measured roofline"
  ]
}

## RQ5

{
  "selection": "Own framework-dispatched32lane kernels possible inall3, but sourcewarp-shuffle semantics require explicitCUDAbackend interop inKA/JACC; performance selection mustinclude microbenchmarkhostoverhead.",
  "application_scope": "Domain-neutral sort primitive, notapplication accuracy validation."
}

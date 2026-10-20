# TPACF observations

## RQ1

{
  "evidence": "All20 trials exact complete final outputs and actual input prefixes; 61440 actual guard bytes/model and all repeated H2D actualframes independently audited. Full integer suffix-multiplicity oracle,90digit cutpoint certificate.",
  "qualified_native_reference": {
    "native_reference_repaired": true,
    "exclude_untouched_native_aggregates": true,
    "native_behavior_known_scientific_defect": true,
    "exclude_from_scientific_fidelity_aggregates": true,
    "reference_provenance": "src/tpacf-julia/reference_provenance.json"
  },
  "limits": "Synthetic unit-sphere catalogs, full source point count97176 with2 rather than100 random catalogs; partial/tiny controls. Not observational astronomy validation or rigorous jackknife uncertainty.",
  "unrepaired_source_policy": [
    "Native one-sided DD group exclusion is not both-endpoint jackknife",
    "Source report overflow indexing retained"
  ]
}

## RQ2

{
  "units": "CUDAjl4complete source bundles,1mathematical GPU body bundle/4definitions,3GPU process checks; KA/JACC1bundle/process each. Backend invocation counts unknown.",
  "feedback": "CUDAjl1CPU syntax rejection,2host orchestration rejects; device mathematics unchanged. Native CPU source witnesses separate from GPU target failures.",
  "translation_tokens": {
    "cuda_jl": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 5650176,
      "output_tokens": 27194,
      "cached_input_tokens": 5371520,
      "reasoning_output_tokens": 13557,
      "total_tokens": 5677370,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T14:55:04.614Z",
        "context_event_sha256": "8f087fcf83fa3fa8fb96fbbd5904b5636a3df77e4a4206ce4439e8a9643ce4ef",
        "source_snapshot_bytes": 125587120,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/tpacf/cuda_jl_translation.jsonl",
      "marker_lag_seconds": 8.270091,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "ka": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 1377959,
      "output_tokens": 6167,
      "cached_input_tokens": 1345664,
      "reasoning_output_tokens": 3269,
      "total_tokens": 1384126,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T14:55:04.614Z",
        "context_event_sha256": "8f087fcf83fa3fa8fb96fbbd5904b5636a3df77e4a4206ce4439e8a9643ce4ef",
        "source_snapshot_bytes": 125789392,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/tpacf/ka_translation.jsonl",
      "marker_lag_seconds": 6.782925,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    },
    "jacc": {
      "status": "observed_declared_activity_window",
      "method": "Separateprospectiveharnesscumulativecounterbegin/end, context-inclusive andnonexclusive; notbilling/exclusivecodegenerationcost",
      "input_tokens": 1302813,
      "output_tokens": 5042,
      "cached_input_tokens": 1255040,
      "reasoning_output_tokens": 2125,
      "total_tokens": 1307855,
      "observed_model_metadata": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "metadata_timestamp": "2026-10-20T15:15:50.345Z",
        "context_event_sha256": "75b84ceb02489b2e31ae82c069336155c96b44a68db37a61e11e97d19680a55a",
        "source_snapshot_bytes": 126384185,
        "lookup_bytes_reviewed": 2097152,
        "lookup_method": "Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.",
        "scope": "Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery."
      },
      "boundary_path": ".porting-state/portable_julia/usage_boundaries_300/tpacf/jacc_translation.jsonl",
      "marker_lag_seconds": 5.964001,
      "scope": "Translation+compilerfeedback+verificationwait inonebenchmark; excludesnativeprep/separatelyscopedannotation. Cachedinput/reasoning aresubsets, notadditionalcosts."
    }
  },
  "scope": "Context-inclusive activity windows, nested verification overlaps; CUDAjl interrupted/reopened and excluded clean cost calibration."
}

## RQ3

{
  "mean_seconds_last4": {
    "cuda": 2.3951572245,
    "cuda_jl": 0.4172188940000002,
    "ka": 0.42856697025000007,
    "jacc": 0.43922926575000004
  },
  "timing_scope": "Sum source tile-compute intervals: input copies, kernels, result copy, CPU partial accumulation; excludes I/O, global setup, report, independent checking and paused actual-input capture. Full warmup outside five trials.",
  "limits": "Native series several hours before ports; no causal abstraction speedup or contemporaneous ranking. Named catalog work and CPU orchestration included."
}

## RQ4

{
  "supported_features": "Own4GPU entry kernels in each model; constant space is genuine AS4 backend-dependent hybrid, not framework unsupported.",
  "confounds": [
    "Native one-sided DD group exclusion is not both-endpoint jackknife",
    "Source report overflow indexing retained",
    "Original sky catalog URL unavailable; synthetic repeated56directions",
    "Native timing separated by hours from ports, no counterbalanced causal comparison",
    "Source fixed16384-square workspace persists even tiny control",
    "Own C++ vs Julia CPU controller inside measured compute region"
  ],
  "no_measured_roofline_claim": true
}

## RQ5

{
  "selection": "KA/JACC can express this tile/histogram pipeline with own kernels and Julia host code, but faithful constant-space mapping needs CUDA LLVM/module interop on this pinned backend.",
  "scope": "No universal cross-vendor/astrophysical accuracy claim."
}

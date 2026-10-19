# FRNA: qualified RNA minimum-energy DP

Six namedcases including originalRD0260 77nt andfive controlled9/12/13/44nt cases, full10entrypoints/allrecurrences. Corrected initialization andrawtablecoefficient convention; noHIVfullCLI, tracebackstructure, partitionfunction, biologicalgroundtruth orfullRNAstructuremodel equivalence claimed.

## RQ1

{
  "fidelity": "Conditionalcorrectedreference exactall4/full10GPUstages; independentglobalMFEcertificate forfivecontrols, 77ntsourcecrosschecknotglobalindependentbiology",
  "original_native_witnesses": "1080uninitreads+CPUInt32OOB, rawtable1.079scale100 vs10 convention. Nativekernelsunchanged."
}

## RQ2

{
  "candidate_attempts": {
    "cuda_jl": 3,
    "ka": 1,
    "jacc": 1
  },
  "actual_usage": {
    "cuda_jl": {
      "input_tokens": 10833261,
      "cached_input_tokens": 10698624,
      "cache_write_input_tokens": 0,
      "output_tokens": 48782,
      "reasoning_output_tokens": 27530,
      "total_tokens": 10882043
    },
    "ka": {
      "input_tokens": 507009,
      "cached_input_tokens": 502656,
      "cache_write_input_tokens": 0,
      "output_tokens": 2713,
      "reasoning_output_tokens": 706,
      "total_tokens": 509722
    },
    "jacc": {
      "input_tokens": 737348,
      "cached_input_tokens": 725760,
      "cache_write_input_tokens": 0,
      "output_tokens": 2564,
      "reasoning_output_tokens": 447,
      "total_tokens": 739912
    }
  },
  "feedback": "CUDAjlhostparse/assignmentexpressionB2; KA/JACC firstpass afterhelperreuse; preparation andCPUoraclerepairs separatelyrecorded."
}

## RQ3

{
  "mean_last_four_seconds": {
    "cuda": 0.008556301249999999,
    "cuda_jl": 0.01010280025,
    "ka": 0.013864035750000002,
    "jacc": 0.02036160375
  },
  "raw_times_preserved": true,
  "timing_scope": "sum164 contextcompleted initialization/perdistancefull9stage segments, allocation/reset/captures/checks outsideROI; fivecheckedtrials mean2\u20135"
}

## RQ4

{
  "mechanism": "Scalar andblockmin DPstages, parameterunits matter despitecommon-modeGPUagreement; sourcepotentialneighborfilter andfiniteINF restrictmodel. Nohardwarecounter/causalrooflineclaim."
}

## RQ5

{
  "selection": "KAexpresseslocalmemory/barriers directly forsourcealgorithm; JACChere usesCUDAbackendextensions toretainstaticsharedtree. Observednamedcase timesnotuniversalframeworkranking. Scientificgate/unitconventions precedeperformancechoice."
}

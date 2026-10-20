# nbnxm observations

## RQ1

{"fidelity": "All20 full captures pass independent analytic pair/Newton and dual references; original input masks zero, active controls mandatory.", "accuracy": "All nine actual device buffers byte-compared against six full native canonical readbacks every trial/model; full force and 45 shift vectors, 2462658 Float64-encoded Float32 values/trial, 1231362 exact-prefix values. Full independent analytic LJ/Ewald pair oracle at .002+1e-4*abs(ref); exact default/tails/untouched fields; normalized Newton<=5e-5; nonfinite rejected.", "domain": "Molecular nonbonded Lennard-Jones and Ewald-correction cluster force operator"}

## RQ2

{"complete_candidates": {"cuda_jl": 1, "ka": 1, "jacc": 1}, "target_compiler_failures": 0, "effort": "effort_summary.json", "limitation": "Ordered reuse, inclusive cached contexts and native prep separate; no intrinsic framework token ranking."}

## RQ3

{"timing_summary": "all4_accuracy_timing_summary.json", "means_seconds": {"cuda": 1.2285976545000001, "cuda_jl": 4.3995751575, "ka": 5.28477516475, "jacc": 5.326934222249999}, "scope": "Sum6resident-data sourcekernellaunchloopspluswholecontextcompletion; separatelyrecordall6casetimes. Original2cases10000repeats, activecontrols1. Allocate/init/prefetch/readback/oracle/serializationoutsideROI. Whole6casewarmupthen5checkedtrialsmean2–5. NativeUMGPUresidentinterventioncommonall4, notoriginalunobservedfirst-touchmigrationlatency."}

## RQ4

{"mechanisms": ["Original 3199 superclusters/204808 atoms/10000 repeats for both shift flags, plus four nonzero/self/exclusion/cutoff controls. Original masks all zero: no physical pair arithmetic in default. Source GPU body unchanged; common explicitly resident managed-data study, not full GROMACS/PME/integrated MD or original first-touch latency.", "Default no-pair launch workload dominates aggregate; active controls repeat once, unsuitable robust physical throughput ranking.", "Host dispatch plus device kernel/completion measured; no device/host causal attribution or hardware profile.", "KA always_inline=true; CUDAjl default compiler controls; shared equations inherit prior CUDA idioms.", "JACC 1.4 CUDAExt 3D launch omits LaunchSpec.stream; actual task legacy stream set explicitly with context completion fences."], "source_derived_not_hardware_measured": {"shared_bytes": 2048, "barriers_per_block": 33, "shift_zero_atomics_default_per_launch": 76776, "decoded_active_control_pairs": 3773}, "hardware_counters": null}

## RQ5

{"selection": "All three support this scoped CUDA-backend hybrid; no backend-independent portability or physical-throughput recommendation from degenerate default.", "support_boundary": "Original 3199 superclusters/204808 atoms/10000 repeats for both shift flags, plus four nonzero/self/exclusion/cutoff controls. Original masks all zero: no physical pair arithmetic in default. Source GPU body unchanged; common explicitly resident managed-data study, not full GROMACS/PME/integrated MD or original first-touch latency."}

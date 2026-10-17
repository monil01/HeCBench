# Matérn initialization control

Diagnostic CUDA-backend-specific shared arena, not a replacement primary JACC implementation. Changes only arena acquisition/initialization: same4192B request and algorithm/launch/inputs/timing, but compiler resource use can also change. All5 dual-reference numerical checks pass and fullartifacts exactly match primary. Primarymean1.726663ms, control.18716725ms, retainedfourtimes averaged. Source/config/proof hashes and attemptscope in verification.json; all5 primarytrials preserved.

Reproduce with tools/verify_portable_julia.py --config src/matern-julia/study_config_jacc_backend_arena_control.json from source root. Inline configuration_role diagnostic_ablation excludes this from primary census summaries even without the ignored research folder. This tests the JACC.shared initialization operator and codegen jointly; not an intrinsic framework limitation or crossbackend result.

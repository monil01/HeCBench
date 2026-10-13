Warp shuffle sort IRIS dmem port
===============================

Retains original register shuffle/bitonic/recursive-merge sorting templates and native HIP logical warp32 adaptation. Export wrappers accept dmem pointers and int length and reconstruct original DeviceTensor arguments inside the GPU kernel. Host sorts allocate iris::DMem only; all keys/indices flush for CPU checks. HIP headers are copied from native adaptation under hip/, original CUDA headers under cuda/; no invented sort/library replacement. Default repeat1000, random-device shuffles and source test shapes unchanged.

Validation runs original repeat10 tests against unchanged isolated native HIP: special +/-infinity, signed zero, subnormal duplicates; sizes16/32/64/128 register sorting; complete output key equality and full index range/input-value/uniqueness checks. All three native PASS verdicts must match IRIS. Random_device permutations are not fixed or compared by index between models; each model checks every value/index against its own input exactly, as original tests require. IRIS fail verdicts also return nonzero.

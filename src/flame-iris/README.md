# Fractal flame with IRIS DMem

Retains all original transform kernels, short/half conversions, struct-by-value
ConstMemParams fields, input RNG seed2/permutations, all dimensions and repeat1000.
IRIS initializes after complete original RNG/input generation; backing input
arrays remain alive for DMem. Host vector layouts match HIPfloat2/float3/short2
(8/12/4bytes). The1928byteparameterstruct exceeds IRIS's1024bytescalar storage,
so it is passed as read-only DMem, updated after each host mapping reset.
Native original is unchanged; isolated adapter adds only optional complete
binary vertex dump HECBENCH_FLAME_OUTPUT, mirrored by IRIS. DUMP printing
semantics retained. Full x/y/color channels are compared at repeat1/default1000.

# MoE token alignment with IRIS DMem

Retains original counting/alignment/sorting device helpers and device-only
CUBBlockScan (hipCUBonAMD), exported through fixedint wrappers. Default token,
expert,top-k,block-size cases and repeat100000 are preserved. MI100HIPwave64
matches nativeHIPhost-query geometry; CUDA kernel retained for future use.
Dynamic counts/small-batch scratch becomes static bounded storage1024/4225ints
covering original expert limits without algorithm substitution. Partial sorted,
expert and cumsum writes use DMemread/write; null map uses pointer-width scalar.
HECBENCH_MOE_OUTPUT optionally dumps every configuration's metadata,inputIDs,
all padded tokens,and expertIDs. Native isolated adapter adds only same dump.
Verification compares complete token multisets per expert (order is atomic and
not stable),padding/counts/expertIDs, and all original30PASSverdicts.
Original small-batch barriers and block counting algorithms remain unchanged.

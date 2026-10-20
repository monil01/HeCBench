# Actual device-runtime feature unit; no full benchmark or performance certificate.
using CUDA
include("c_memory.jl")
function probe_heap!(out)
    i=Int32(threadIdx().x)
    p=CUDA.malloc(Csize_t(28))
    if p==C_NULL;out[i]=Int32(-1);return;end
    c=LRPtr{LRCut}(p)[0];c.sign=i*Int32(17);c.used=true;c.truthtable=UInt32(0x1234);c.nLeaves=UInt32(3);c.value=UInt32(1001)
    out[i]=c.sign+Int32(c.truthtable)+Int32(c.nLeaves)+Int32(c.value)+Int32(c.used)
    lr_free(p)
    return
end
CUDA.device!(0);out=CUDA.zeros(Int32,32)
CUDA.@cuda threads=32 blocks=1 always_inline=true probe_heap!(out)
CUDA.device_synchronize();got=Array(out)
@assert got==Int32[17*i+0x1234+3+1001+1 for i in 1:32]
println("UNIT_DEVICE_HEAP_PASS ",got)

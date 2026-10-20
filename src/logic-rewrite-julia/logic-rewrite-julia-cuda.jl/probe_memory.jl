# Translation-unit device test only; not a full benchmark/timing certificate.
using CUDA
include("c_memory.jl")
function probe_memory!(raw,inputs,out)
    i=Int32(threadIdx().x)
    if i==1
        p=LRPtr{LRCut}(pointer(raw));c=p[0]
        c.sign=Int32(-12345)
        leaves=c.leaves
        for k in Int32(0):Int32(3);leaves[k]=Int32(7)+k;end
        c.truthtable=UInt32(0xaaaa);c.value=UInt32(1001);c.nLeaves=UInt32(4);c.used=true
        for j in Int32(1):Int32(length(inputs));out[j]=lr_countones(inputs[j]);end
    end
    return
end
CUDA.device!(0)
raw=CUDA.zeros(UInt8,28);inputs=CuArray(UInt32[0,1,3,0x55555555,0xffffffff]);out=CUDA.zeros(Int32,5)
CUDA.@cuda threads=1 blocks=1 always_inline=true probe_memory!(raw,inputs,out)
CUDA.device_synchronize()
got=Array(raw);counts=Array(out)
@assert reinterpret(Int32,got[1:4])[1]==-12345
@assert reinterpret(Int32,got[5:20])==Int32[7,8,9,10]
word=reinterpret(UInt32,got[21:24])[1]
@assert word==UInt32(0xaaaa)|(UInt32(1001)<<16)|(UInt32(4)<<27)
@assert got[25]==1
@assert counts==Int32[0,1,2,16,32]
println("UNIT_DEVICE_PASS ",repr((cut_bytes=got,popcounts=counts)))

# Source-derived helper candidate unit, NOT complete benchmark verification.
using CUDA
include("c_memory.jl")
include("generated_cut_helpers.jl")
function probe_generated!(raw,refs,ctx,n)
    lr_Inputs(ctx,LRPtr{Int32}(pointer(refs)),LRPtr{LRCut}(pointer(raw)),n)
    return
end
CUDA.device!(0);n=Int32(6);raw=CUDA.zeros(UInt8,(Int(n)+1)*8*28);refs=CuArray(Int32[0,1,2,3,4,5,6]);globals=CuArray(Int32[17,7])
# Host pointers cannot be LRPtr payload to a GPU: the wrappers' CuDeviceArray
# conversion must occur inside device kernel for actual device address spaces.
function run_inputs!(raw,refs,globals,n)
    ctx=(P=LRPtr{Int32}(pointer(globals)),N=LRPtr{Int32}(pointer(globals)+sizeof(Int32)))
    probe_generated!(raw,refs,ctx,n)
    return
end
CUDA.@cuda threads=64 blocks=1 always_inline=true run_inputs!(raw,refs,globals,n)
CUDA.device_synchronize();got=Array(raw)
for i in 1:Int(n)
    offset=i*8*28
    @assert reinterpret(Int32,got[offset+1:offset+4])[1]==Int32(1)<<(i&31)
    @assert reinterpret(Int32,got[offset+5:offset+8])[1]==i
    word=reinterpret(UInt32,got[offset+21:offset+24])[1]
    @assert word&0xffff==0xaaaa
    @assert (word>>27)&15==1
    @assert (word>>16)&0x7ff==1001
    @assert got[offset+25]==1
end
println("UNIT_GENERATED_INPUTS_PASS six source input cuts; fullpipeline remains pending")

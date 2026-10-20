# Full source cut-enumeration subsystem unit against original CUDA outputs.
# Does not certify the balance/refactor/strash/full resyn2 benchmark pipeline.
using CUDA
include("c_memory.jl")
include("generated_cut_helpers.jl")
function run_convert!(a,c,n)
    lr_Convert(nothing,LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(c)),n)
    return
end
function run_enum!(a,b,c0,c1,refs,cuts,delta,n)
    lr_CutEnumerate(nothing,LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(b)),LRPtr{Int32}(pointer(c0)),LRPtr{Int32}(pointer(c1)),LRPtr{Int32}(pointer(refs)),LRPtr{LRCut}(pointer(cuts)),delta,n)
    return
end
function run_inputs!(refs,cuts,n)
    lr_Inputs(nothing,LRPtr{Int32}(pointer(refs)),LRPtr{LRCut}(pointer(cuts)),n)
    return
end
f=open(ARGS[1]);n=read(f,Int32);np=read(f,Int32);a=read!(f,Vector{Int32}(undef,n+1));b=read!(f,similar(a));refs=read!(f,similar(a));expected=read(f);close(f)
CUDA.device!(0)
da=CuArray(a);db=CuArray(b);dr=CuArray(refs);c0=CUDA.zeros(Int32,n+1);c1=CUDA.zeros(Int32,n+1);cuts=CUDA.zeros(UInt8,(n+1)*8*28)
@assert Array(da)==a && Array(db)==b && Array(dr)==refs
CUDA.@cuda threads=512 blocks=1 always_inline=true run_convert!(da,c0,n+Int32(1))
CUDA.@cuda threads=512 blocks=1 always_inline=true run_convert!(db,c1,n+Int32(1))
CUDA.@cuda threads=512 blocks=1 always_inline=true run_inputs!(dr,cuts,np)
for j in np+Int32(1):n
    CUDA.@cuda threads=512 blocks=1 always_inline=true run_enum!(da,db,c0,c1,dr,cuts,j-Int32(1),Int32(1))
    CUDA.device_synchronize()
end
CUDA.device_synchronize();got=Array(cuts)
write(ARGS[2],got)
if got!=expected
    ix=findall(got.!=expected)
    println("CUT_UNIT_DIFFERENCES ",length(ix)," first=",ix[1:min(end,20)])
    for k in ix[1:min(end,20)];println("byte ",k," actual=",got[k]," expected=",expected[k]);end
    error("full source cut-enumeration raw output mismatch")
end
println("UNIT_SOURCE_CUT_ENUMERATION_PASS ",length(got)," raw bytes, ",n+1," nodes, four inputs")

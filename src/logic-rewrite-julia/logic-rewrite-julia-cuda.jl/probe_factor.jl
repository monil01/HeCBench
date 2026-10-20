# Selected original factorFromTruth algorithm unit: all 256 3-input functions.
# Not a full refactor/fullresyn2 benchmark or performance certificate.
using CUDA
include("c_memory.jl");include("strash_helpers.jl");include("factor_helpers.jl");include("generated_factor.jl")
function factor_entry!(cuts,ranges,table,links,lens,next,truth,neg,truthranges,elem,n)
    lr_factorFromTruth(nothing,LRPtr{Int32}(pointer(cuts)),LRPtr{Int32}(pointer(ranges)),LRPtr{UInt64}(pointer(table)),LRPtr{Int32}(pointer(links)),LRPtr{Int32}(pointer(lens)),LRPtr{Int32}(pointer(next)),LRPtr{UInt32}(pointer(truth)),LRPtr{UInt32}(pointer(neg)),LRPtr{Int32}(pointer(truthranges)),LRPtr{UInt32}(pointer(elem)),n);return nothing
end
f=open(ARGS[1]);n=read(f,Int32);cuts=read!(f,Vector{Int32}(undef,3*n));ranges=read!(f,Vector{Int32}(undef,n));truthranges=read!(f,Vector{Int32}(undef,n));truth=read!(f,Vector{UInt32}(undef,n));neg=read!(f,Vector{UInt32}(undef,n));elem=read!(f,Vector{UInt32}(undef,1));expected=read(f);close(f)
CUDA.device!(0);CUDA.context();CUDA.CUDACore.cuCtxSetLimit(CUDA.CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE,256*1024*1024);CUDA.CUDACore.cuCtxSetLimit(CUDA.CUDACore.CU_LIMIT_STACK_SIZE,65536)
dc=CuArray(cuts);dr=CuArray(ranges);dtr=CuArray(truthranges);dt=CuArray(truth);dn=CuArray(neg);de=CuArray(elem);table=CuArray(zeros(UInt64,4*n*8));links=CuArray(fill(Int32(-1),4*n));lens=CuArray(zeros(Int32,2*n));next=CuArray(Int32[2*n])
@assert Array(dc)==cuts&&Array(dr)==ranges&&Array(dtr)==truthranges&&Array(dt)==truth&&Array(dn)==neg&&Array(de)==elem
CUDA.@cuda threads=512 blocks=1 maxregs=128 factor_entry!(dc,dr,table,links,lens,next,dt,dn,dtr,de,n)
CUDA.device_synchronize();out=IOBuffer();for a in (table,links,lens,next);write(out,Array(a));end;got=take!(out);write(ARGS[2],got)
if got!=expected
    ix=findall(got.!=expected);println("FACTOR_UNIT_RAW_MISMATCH ",length(ix)," first=",ix[1:min(end,24)]);error("sourceISOP/factorunit differsfromoriginalCUDA")
end
println("UNIT_ORIGINAL_FACTOR_ALL256_THREE_INPUT_FUNCTIONS_PASS rawbytes=",length(got)," fullpipeline_pending=true")

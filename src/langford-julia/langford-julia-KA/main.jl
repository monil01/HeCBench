using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function kernel(result,v::Val{N}) where N
 lid=@index(Local,Linear);gid=@index(Global,Linear)
 avail=@localmem Int32 (4*(2N+1),)
 opens=@localmem Int64 (4*(4N+2),)
 stack=@localmem Int8 (4*24N,)
 pos=@localmem Int8 (4*8*((N+7)÷8),)
 dfs_lane!(result,avail,opens,stack,pos,lid,gid,v)
end
langford_study((result,v)->begin;k=kernel(CUDABackend(),4);()->k(result,v;ndrange=16384);end)

using CUDA
include("../study_common.jl")
function kernel(result,v::Val{N}) where N
 lid=threadIdx().x;gid=(blockIdx().x-1)*blockDim().x+lid
 avail=CuStaticSharedArray(Int32,4*(2N+1));opens=CuStaticSharedArray(Int64,4*(4N+2));stack=CuStaticSharedArray(Int8,4*24N);pos=CuStaticSharedArray(Int8,4*8*((N+7)÷8))
 dfs_lane!(result,avail,opens,stack,pos,lid,gid,v)
 return
end
langford_study((result,v)->()->(@cuda threads=4 blocks=4096 kernel(result,v)))

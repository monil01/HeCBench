using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function kreduce_cuda!(input,out,n::Int32,mode)
 tid=threadIdx().x;block=blockIdx().x;i=(block-Int32(1))*Int32(256)+tid;tile=CuStaticSharedArray(KResult,256);@inbounds tile[tid]=i<=n ? kload(input,i,mode) : zero(KResult);sync_threads()
 stride=Int32(128)
 while stride>0
  if tid<=stride;@inbounds tile[tid]=kmerge(tile[tid],tile[tid+stride]);end
  sync_threads();stride>>=1
 end
 if tid==1;@inbounds out[block]=tile[1];end
 return
end
function launch!(x,partial,out,n,B)
 @cuda threads=256 blocks=B kreduce_cuda!(x,partial,Int32(n),Val(0))
 @cuda threads=256 blocks=1 kreduce_cuda!(partial,out,Int32(B),Val(1))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
kurtosis_study(launch!)

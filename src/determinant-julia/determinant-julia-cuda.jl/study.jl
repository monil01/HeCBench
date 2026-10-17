using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function diag_product!(a,out)
 tid=Int32(threadIdx().x);scratch=CuStaticSharedArray(Float32,16)
 @inbounds scratch[tid]=tid<=Int32(11) ? a[(tid-Int32(1))*Int32(12)+Int32(1)] : 1f0
 sync_threads();stride=Int32(8)
 while stride>0
  if tid<=stride;@inbounds scratch[tid]*=scratch[tid+stride];end
  sync_threads();stride=div(stride,Int32(2))
 end
 if tid==Int32(1);@inbounds out[1]=scratch[1];end
 return
end
factory(a,out)=()->(@cuda threads=16 blocks=1 diag_product!(a,out))
determinant_study(factory)

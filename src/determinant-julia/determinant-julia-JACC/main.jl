using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function diag_product!(index,template,a,out)
 tid=Int32(index);scratch=JACC.shared(template)
 @inbounds scratch[tid]=tid<=Int32(11) ? a[(tid-Int32(1))*Int32(12)+Int32(1)] : 1f0
 JACC.sync_workgroup();stride=Int32(8)
 while stride>0
  if tid<=stride;@inbounds scratch[tid]*=scratch[tid+stride];end
  JACC.sync_workgroup();stride=div(stride,Int32(2))
 end
 if tid==Int32(1);@inbounds out[1]=scratch[1];end
 return
end
function factory(a,out)
 template=CUDA.ones(Float32,16);spec=JACC.launch_spec(;threads=16,shmem_size=64,sync=false)
 return ()->JACC.parallel_for(spec,16,diag_product!,template,a,out)
end
determinant_study(factory)

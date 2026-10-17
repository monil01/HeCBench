using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function diag_product!(a,out)
 tid=Int32(@index(Local,Linear));scratch=@localmem Float32 (16,)
 @inbounds scratch[tid]=tid<=Int32(11) ? a[(tid-Int32(1))*Int32(12)+Int32(1)] : 1f0
 @synchronize;stride=Int32(8)
 while stride>0
  if tid<=stride;@inbounds scratch[tid]*=scratch[tid+stride];end
  @synchronize;stride=div(stride,Int32(2))
 end
 if tid==Int32(1);@inbounds out[1]=scratch[1];end
end
function factory(a,out)
 kernel=diag_product!(get_backend(a),16)
 return ()->kernel(a,out;ndrange=16)
end
determinant_study(factory)

using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(output,input,weights,indices,::Val{BS},::Val{MODE}) where {BS,MODE}
 tid=@index(Local,Linear)
 group=@index(Group,NTuple)
 token=group[1]-Int32(1);channel=group[2]-Int32(1)
 tmp=@localmem Float32 (BS,)
 @inbounds tmp[tid]=partial(tid-Int32(1),token,channel,input,weights,indices,Val(BS),Val(MODE))
 @synchronize
 stride=Int32(BS÷2)
 while stride>0
  if tid<=stride;@inbounds tmp[tid]+=tmp[tid+stride];end
  @synchronize
  stride÷=Int32(2)
 end
 if tid==1;@inbounds store_result(output,token,channel,tmp[1],Val(MODE));end
end
run!(out,input,w,indices,tmp,::Val{BS},::Val{MODE}) where {BS,MODE}=(kernel!(get_backend(out),(BS,1))(out,input,w,indices,Val(BS),Val(MODE);ndrange=(BS*TT,MODE<=2 ? RR : HH));nothing)
study(run!)

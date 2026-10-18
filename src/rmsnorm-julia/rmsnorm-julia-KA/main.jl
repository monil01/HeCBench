using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function rms_ka!(x,xv,g,gv,o,ov,H::Int32,bt::Val{BT},u::Val{U}) where {BT,U}
 tid=Int32(@index(Local,Linear));row=Int32(@index(Group,Linear));warp=(tid-Int32(1))>>5;lane=(tid-Int32(1))&Int32(31);sh=@localmem Float32 (32,);sum=rms_partial(x,xv,row,tid,H,bt,u)
 if lane==0;@inbounds sh[warp+1]=sum;end
 @synchronize
 if warp==0
  value=lane<Int32(BT÷32) ? (@inbounds sh[lane+1]) : 0f0;value=rms_warp_sum(value)
  if lane==0;@inbounds sh[1]=value;end
 end
 @synchronize
 @inbounds total=sh[1];rms_finish!(x,xv,g,gv,o,ov,row,tid,H,total,bt,u)
end
function factory(x,xv,g,gv,o,ov,H,N,bt,u)
 BT=typeof(bt).parameters[1];k=rms_ka!(CUDABackend(),(BT,1))
 ()->k(x,xv,g,gv,o,ov,Int32(H),bt,u;ndrange=(BT,N))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rms_study(factory)

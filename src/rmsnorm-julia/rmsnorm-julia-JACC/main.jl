using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
const RMS_TEMPLATE=JACC.zeros(Float32,32)
function rms_jacc!(ix,iy,x,xv,g,gv,o,ov,H::Int32,bt::Val{BT},u::Val{U},template) where {BT,U}
 tid=Int32(ix);row=Int32(iy);warp=(tid-Int32(1))>>5;lane=(tid-Int32(1))&Int32(31);sh=JACC.shared(template);sum=rms_partial(x,xv,row,tid,H,bt,u)
 if lane==0;@inbounds sh[warp+1]=sum;end
 JACC.sync_workgroup()
 if warp==0
  value=lane<Int32(BT÷32) ? (@inbounds sh[lane+1]) : 0f0;value=rms_warp_sum(value)
  if lane==0;@inbounds sh[1]=value;end
 end
 JACC.sync_workgroup()
 @inbounds total=sh[1];rms_finish!(x,xv,g,gv,o,ov,row,tid,H,total,bt,u)
 return
end
function factory(x,xv,g,gv,o,ov,H,N,bt,u)
 BT=typeof(bt).parameters[1];spec=JACC.launch_spec(;threads=(BT,1),blocks=(1,N),shmem_size=128,sync=false)
 ()->JACC.parallel_for(spec,(BT,N),rms_jacc!,x,xv,g,gv,o,ov,Int32(H),bt,u,RMS_TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rms_study(factory)

using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function rms_cuda!(x,xv,g,gv,o,ov,H::Int32,bt::Val{BT},u::Val{U}) where {BT,U}
 tid=threadIdx().x;row=blockIdx().x;warp=(tid-Int32(1))>>5;lane=(tid-Int32(1))&Int32(31);sh=CuStaticSharedArray(Float32,32);sum=rms_partial(x,xv,row,tid,H,bt,u)
 if lane==0;@inbounds sh[warp+1]=sum;end
 sync_threads()
 if warp==0
  value=lane<Int32(BT÷32) ? (@inbounds sh[lane+1]) : 0f0;value=rms_warp_sum(value)
  if lane==0;@inbounds sh[1]=value;end
 end
 sync_threads();@inbounds total=sh[1];rms_finish!(x,xv,g,gv,o,ov,row,tid,H,total,bt,u)
 return
end
function factory(x,xv,g,gv,o,ov,H,N,bt,u)
 kernel=@cuda launch=false rms_cuda!(x,xv,g,gv,o,ov,Int32(H),bt,u)
 ()->kernel(x,xv,g,gv,o,ov,Int32(H),bt,u;threads=typeof(bt).parameters[1],blocks=N)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rms_study(factory)

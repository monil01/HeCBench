using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function lomb_jacc!(index,ns::Int32,nf::Int32,x,y,freqs,out,y_dot::Float32,launch_stride::Int32)
 idx=Int32(index)-Int32(1);stride=launch_stride
 while idx<nf
  freq=@inbounds freqs[idx+Int32(1)];xc=0.f0;xs=0.f0;cc=0.f0;ss=0.f0;cs=0.f0
  for j in Int32(1):ns
   s,c=sincos(freq*(@inbounds x[j]));value=@inbounds y[j]
   xc+=value*c;xs+=value*s;cc+=c*c;ss+=s*s;cs+=c*s
  end
  tau=atan(2.f0*cs,cc-ss)/(2.f0*freq);s_tau,c_tau=sincos(freq*tau);c_tau2=c_tau*c_tau;s_tau2=s_tau*s_tau;cs_tau=2.f0*c_tau*s_tau
  @inbounds out[idx+Int32(1)]=(0.5f0*(((c_tau*xc+s_tau*xs)*(c_tau*xc+s_tau*xs)/(c_tau2*cc+cs_tau*cs+s_tau2*ss))+((c_tau*xs-s_tau*xc)*(c_tau*xs-s_tau*xc)/(c_tau2*ss-cs_tau*cs+s_tau2*cc))))*y_dot
  idx+=stride
 end
 return
end
function factory(x,y,freqs,out,scale,ns,nf,blocks)
 total=blocks*256;spec=JACC.launch_spec(;threads=256,blocks=blocks,shmem_size=0,sync=false)
 return ()->JACC.parallel_for(spec,total,lomb_jacc!,Int32(ns),Int32(nf),x,y,freqs,out,scale,Int32(total))
end
lomb_study(factory)

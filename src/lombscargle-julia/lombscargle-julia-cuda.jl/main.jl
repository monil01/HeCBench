using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function lomb_kernel!(ns::Int32,nf::Int32,x,y,freqs,out,y_dot::Float32)
 idx=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);stride=blockDim().x*gridDim().x
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
 return ()->(@cuda threads=256 blocks=blocks lomb_kernel!(Int32(ns),Int32(nf),x,y,freqs,out,scale))
end
if abspath(PROGRAM_FILE)==@__FILE__;lomb_study(factory);end

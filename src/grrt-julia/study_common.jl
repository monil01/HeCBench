using CUDA,Printf
include(joinpath(@__DIR__,"grrt_math.jl"))
function study_main(launch!)
 input1=Float64[0,ccall(:acos,Cdouble,(Cdouble,),.25)/GR_PI*180.0,8,340e9];input2=Float64[0,45,8,340e9];inputs=[input1,input2];gold=[zeros(Float64,192) for _ in 1:2]
 for f in 1:2,y in 0:7,x in 0:7;gr_pixel!(gold[f],inputs[f],Int32(x),Int32(y),f==1 ? Val(1) : Val(2));end
 println("STUDY_CPU_GOLD [",join(vcat(gold...),","),"]")
 # Independent Schwarzschild null-Hamiltonian identity at initialization.
 nullmax=0.0
 for f in 1:2,y in 0:7,x in 0:7
  V=MVector{9,Float64}(undef);state=MVector{6,Float64}(undef);d=MVector{6,Float64}(undef);gr_setup!(V,inputs[f],Int32(x),Int32(y),f==1 ? 55.0 : 10.0);gr_initial!(V,inputs[f],state,d)
  rr=state[1];ss=sin(state[2]);h=-1/(1-2/rr)+(1-2/rr)*state[5]^2+state[6]^2/rr^2+V[6]^2/(rr^2*ss^2);nullmax=max(nullmax,abs(h))
 end
 @assert nullmax<1e-10
 dv=CuArray.(inputs);outs=[CUDA.fill(-123.25,192) for _ in 1:2];launch!(outs[1],dv[1],Val(1));launch!(outs[2],dv[2],Val(2));CUDA.synchronize()
 for trial in 1:5
  for o in outs;fill!(o,-123.25);end;CUDA.synchronize();st=time_ns();launch!(outs[1],dv[1],Val(1));launch!(outs[2],dv[2],Val(2));CUDA.synchronize();elapsed=(time_ns()-st)*1e-9
  got=Array.(outs);ok=true;err=0.0;redabs=0.0;lumrel=0.0
  for f in 1:2,i in 1:192
   a=got[f][i];ref=gold[f][i];e=abs(a-ref);err=max(err,e);ok&=isfinite(a)&&e<=1e-8+1e-6*abs(ref)
   if i%3==0
    if f==1;redabs=max(redabs,e);ok&=(a>0)==(ref>0);else;lumrel=max(lumrel,e/abs(ref));ok&=a>0;end
   end
  end
  art=vcat(input1,input2,got...);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":384,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"redshift_max_abs_error\":%.17g,\"luminosity_max_rel_error\":%.17g,\"initial_null_residual_CPU\":%.17g,\"timing_region\":\"both_complete_adaptive_geodesic_and_transfer_images8x8\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":392,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",err,lumrel,redabs,lumrel,nullmax,path)
 end
end

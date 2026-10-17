using CUDA,Printf
const CAVITY_LENGTHS=(1156,1156,1156,1156,612,612,32,32,32,32)
function cavity_compute(launch!,state;report=false,trial=0)
 maxu=2.0;maxv=1e-10;t=0.;dt=0.;norm=0.;iter=0;steps=0
 while t<.02
  dt=.5*min(.5*1000/(2/(1/32)^2),(1/32)/maxu,(1/32)/maxv);if t+dt>=.02;dt=.02-t;end
  launch!(:calculate_F,dt,state);launch!(:calculate_G,dt,state);launch!(:sum_pressure,dt,state)
  ps=Array(state[7]);p0=0.;for x in ps;p0+=x;end;p0=sqrt(p0/1024);if p0<.0001;p0=1.;end
  for k=1:1000000
   iter=k;launch!(:set_horz_pres_BCs,dt,state);launch!(:set_vert_pres_BCs,dt,state);launch!(:red_kernel,dt,state);launch!(:black_kernel,dt,state);launch!(:calc_residual,dt,state)
   residual=Array(state[8]);norm=0.;for x in residual;norm+=x;end;norm=sqrt(norm/1024)/p0;if norm<.001;break;end
  end
  iter<1000000||error("Pressure solver nonconvergence")
  launch!(:calculate_u,dt,state);mu=Array(state[9]);launch!(:calculate_v,dt,state);mv=Array(state[10]);maxu=1e-10;maxv=1e-10
  for i=1:32;maxu=max(maxu,mu[i]);maxv=max(maxv,mv[i]);end
  launch!(:set_BCs,dt,state);t+=dt;steps+=1
  if report;@printf("CAVITY_STEP trial=%d step=%d time=%.12g dt=%.12g iterations=%d residual=%.12g\n",trial,steps,t,dt,iter,norm);end
 end
 return Float64[iter,dt,norm,steps]
end
function cavity_study(launch!)
 prefix=reinterpret(Float64,read(joinpath(@__DIR__,"study_input_prefix.bin")));gold=reinterpret(Float64,read(joinpath(@__DIR__,"study_expected.bin")));initial=Vector{Vector{Float64}}();pos=9
 for i=1:10
  n=CAVITY_LENGTHS[i];push!(initial,i<=6 ? collect(prefix[pos:pos+n-1]) : zeros(n));if i<=6;pos+=n;end
 end
 state=map(CuArray,initial);cavity_compute(launch!,state;report=true,trial=0);CUDA.synchronize();allok=true
 for trial=1:5
  for i=1:10;copyto!(state[i],initial[i]);end;CUDA.synchronize();start=time_ns();metadata=cavity_compute(launch!,state;report=true,trial=trial);CUDA.synchronize();sec=(time_ns()-start)*1e-9
  vals=vcat(map(Array,state)...,metadata);err=maximum(abs.(vals.-gold));ok=all(isfinite,vals)&&all(abs(v-g)<=1e-9+1e-10*abs(g) for (v,g) in zip(vals,gold));allok&=ok
  out=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":5980,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"coupled_transient_projection_loop_host_checks_and_partial_transfers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",err,path,length(out))
 end
 allok||error("Full-grid independent cavity oracle mismatch");println("PASS")
end

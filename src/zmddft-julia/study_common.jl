using Printf
const BASE=@__DIR__;const N=33554432;const G=16;const SENT=12345.25;const REPEATS=100
function readf(name)
 a=collect(reinterpret(Float64,read(joinpath(BASE,name))));a
end
function inspect_outputs(buffers,oracles)
 ok=true;maxabs=0.;maxbudget=0.;guards=Float64[]
 for(q,o)in zip(buffers,oracles)
  a=Array(q);v=view(a,1:N);dif=abs.(v.-o);budget=1e-9 .+ 1e-12 .* abs.(o);ok &=all(isfinite,a)&&all(dif.<=budget);maxabs=max(maxabs,maximum(dif));maxbudget=max(maxbudget,maximum(dif./budget));tail=a[N+1:end];ok &=all(==(SENT),tail);append!(guards,tail)
 end
 ok,maxabs,maxbudget,guards
end
function zmddft_study(model,makearray,factory)
 input=readf("input_complex_f64.bin");length(input)==N||error("Inputlength");tw=readf("twiddles_f64.bin");oracles=[readf("oracle_stage$(i)_f64.bin") for i=0:2];D3=makearray(tw);X=makearray(fill(SENT,N+G));P1=makearray(fill(SENT,N+G));P2=makearray(fill(SENT,N+G));Y=makearray(fill(SENT,N+G));launch=factory(X,P1,P2,Y,D3)
 impulse=zeros(Float64,N+G);impulse[N+1:end].=SENT;idx=2*(3*256*256+5*256+7)+1;impulse[idx]=.75;impulse[idx+1]=-.5;copyto!(X,impulse)
 reset=fill(SENT,N+G);reset[1:N].=NaN
 for b in (P1,P2,Y);copyto!(b,reset);end
 launch();CUDA.device_synchronize(blocking=true)
 diagdir=joinpath(BASE,"../../.porting-state/portable_julia/diagnostics/zmddft");mkpath(diagdir);diagpath=joinpath(diagdir,"$(model)_impulse_stages_f64.bin");warmok=true;warmmax=0.;warmbudget=0.
 open(diagpath,"w")do io
  for(stage,b)in enumerate((P1,P2,Y))
   a=Array(b);o=readf("impulse_stage$(stage-1)_f64.bin");dif=abs.(view(a,1:N).-o);budget=1e-9 .+ 1e-12 .* abs.(o);warmok &=all(isfinite,a)&&all(dif.<=budget)&&all(==(SENT),a[N+1:end]);warmmax=max(warmmax,maximum(dif));warmbudget=max(warmbudget,maximum(dif./budget));write(io,reinterpret(UInt8,a[1:N]))
  end
 end
 warmok &=Array(X)==impulse&&Array(D3)==tw
 open(joinpath(diagdir,"$(model)_impulse.json"),"w")do io
  @printf(io,"{\"model\":\"%s\",\"correct\":%s,\"timed\":false,\"location\":[3,5,7],\"amplitude\":[0.75,-0.5],\"all3stage_values\":%d,\"max_abs_error\":%.17g,\"max_normalized_budget_error\":%.17g,\"stage_artifact\":\"%s\"}\n",model,warmok ? "true" : "false",3*N,warmmax,warmbudget,diagpath)
 end
 warmok||error("Analyticfullstageimpulsecontrolfailed")
 copyto!(X,vcat(input,fill(SENT,G)));launch();CUDA.device_synchronize(blocking=true);ok,ma,mb,gg=inspect_outputs((P1,P2,Y),oracles);ok||error("PrimarywarmupCPUFFTfailed")
 for trial=1:5
  for b in (P1,P2,Y);copyto!(b,reset);end;CUDA.device_synchronize(blocking=true);start=time_ns();for rep=1:REPEATS;launch();end;CUDA.device_synchronize(blocking=true);elapsed=(time_ns()-start)*1e-9
  actualX=Array(X);actualTW=Array(D3);ok=actualX==vcat(input,fill(SENT,G))&&actualTW==tw;maxabs=0.;maxbudget=0.;guards=collect(actualX[N+1:end]);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin")
  open(path,"w")do io
   write(io,reinterpret(UInt8,actualX[1:N]));write(io,reinterpret(UInt8,actualTW));write(io,reinterpret(UInt8,Float64[256,256,256,REPEATS,64,16384,3,512]))
   for(b,o)in zip((P1,P2,Y),oracles)
    a=Array(b);v=view(a,1:N);dif=abs.(v.-o);budget=1e-9 .+ 1e-12 .* abs.(o);ok &=all(isfinite,a)&&all(dif.<=budget);maxabs=max(maxabs,maximum(dif));maxbudget=max(maxbudget,maximum(dif./budget));tail=a[N+1:end];ok &=all(==(SENT),tail);append!(guards,tail);write(io,reinterpret(UInt8,a[1:N]))
   end
   write(io,reinterpret(UInt8,guards))
  end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_budget_error\":%.17g,\"timing_region\":\"R100_full256cube_F64_Spiral3FFTstages_devicecomplete\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",3*N+4*G,maxabs,maxbudget,path,4*N+512+8+4*G);flush(stdout);ok||error("Full3stageFFT/inputs/guardsfailed")
 end
end

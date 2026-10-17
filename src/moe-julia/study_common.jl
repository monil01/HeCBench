using CUDA,Printf
function moe_study(launch!)
 input=reinterpret(Float32,read("moe_input_float32.bin"));gold=reinterpret(Float64,read("moe_expected_float64.bin"));states=[];prefix=Float64[];offset=0
 for pattern=0:1,k in (1,2,4,8)
  a=copy(input[pattern*98688+1:(pattern+1)*98688]);n=257*k;count=98688+3*n
  oracle=copy(gold[offset+1:offset+count]);offset+=count
  da=CuArray(a);p=CUDA.zeros(Float32,98688);w=CUDA.zeros(Float32,n);i=CUDA.zeros(Int32,n);r=CUDA.zeros(Int32,n)
  append!(prefix,Float64[pattern,k,257,384]);append!(prefix,Float64.(a));append!(prefix,zeros(n))
  for _=1:2;launch!(da,p,w,i,r,k);end
  push!(states,(k,da,p,w,i,r,oracle))
 end
 @assert offset==812634 && length(prefix)==797246
 CUDA.device_synchronize(blocking=true)
 for trial=1:5
  for (k,a,p,w,i,r,g) in states;fill!(i,0);end
  CUDA.device_synchronize(blocking=true);times=Float64[]
  for (k,a,p,w,i,r,g) in states
   start=time_ns()
   for _=1:20;launch!(a,p,w,i,r,k);end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/20)
  end
  out=copy(prefix);ma=0.0;mr=0.0;ok=true
  for (k,a,p,w,i,r,g) in states
   prob=Float64.(Array(p));weights=Float64.(Array(w));idx=Float64.(Array(i));rows=Float64.(Array(r));n=257*k;got=vcat(prob,weights,idx,rows);append!(out,got)
   err=abs.(got[1:98688+n].-g[1:98688+n]);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(g[1:98688+n]),1e-6)))
   ok&=all(isfinite,got)&&all(err.<=1e-6.+1e-5.*abs.(g[1:98688+n]))&&idx==g[98688+n+1:98688+2*n]&&rows==g[98688+2*n+1:end]
   ok&=all(0 .<=prob.<=1)&&all(abs(sum(prob[t*384+1:(t+1)*384])-1)<=1e-5 for t=0:256)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%s],\"correct\":%s,\"checked_values\":812634,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"relative_error_denominator_floor\":0.000001,\"timing_region\":\"eight_routing_case_twokernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),join(times,","),ok ? "true" : "false",ma,mr,path,length(out))
  ok||error("Fullroutingoraclefailed")
 end
 println("PASS")
end

using CUDA,Printf
const ATTN_NS=(1,3,65,127)
const ATTN_TYPES=(Float16,CUDA.CUDACore.BFloat16)
function attn_inputs(T,n)
 q=T.([Float32(mod(7i+3,31)-15)/16f0 for i=0:2047])
 k=T.([Float32(mod(11i+5,29)-14)/16f0 for i=0:n*256-1])
 v=T.([Float32(mod(13i+7,27)-13)/16f0 for i=0:n*256-1])
 q,k,v
end
function attn_gold(q,k,v,n)
 out=zeros(Float64,2048)
 for h=0:31
  kh=h÷8
  logits=[sum(Float64(q[h*64+c+1])*Float64(k[s*256+kh*64+c+1]) for c=0:63)/8 for s=0:n-1]
  weights=exp.(logits.-maximum(logits));weights./=sum(weights)
  for c=0:63;out[h*64+c+1]=sum(weights[s+1]*Float64(v[s*256+kh*64+c+1]) for s=0:n-1);end
 end
 out
end
function attention_study(launch!)
 states=[];prefix=Float32[]
 for (kind,T) in enumerate(ATTN_TYPES),n in ATTN_NS
  q,k,v=attn_inputs(T,n);gold=attn_gold(q,k,v,n)
  append!(prefix,Float32[kind,n,32,4,64]);append!(prefix,Float32.(q));append!(prefix,Float32.(k));append!(prefix,Float32.(v))
  dq=CuArray(q);dk=CuArray(k);dv=CuArray(v);o=CUDA.zeros(T,2048)
  for _=1:2;launch!(dq,dk,dv,o,n);end
  push!(states,(n,dq,dk,dv,o,gold))
 end
 CUDA.device_synchronize(blocking=true)
 for trial=1:5
  times=Float64[]
  for (n,q,k,v,o,gold) in states
   start=time_ns()
   for _=1:20;launch!(q,k,v,o,n);end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/20)
  end
  out=copy(prefix);ok=true;ma=0.0;mr=0.0
  for (n,q,k,v,o,gold) in states
   got=Float64.(Array(o));append!(out,Float32.(got));err=abs.(got.-gold)
   ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(gold),0.002)))
   ok&=all(isfinite,got)&&all(err.<=0.002.+0.01.*abs.(gold))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%s],\"correct\":%s,\"checked_values\":16384,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"relative_error_denominator_floor\":0.002,\"timing_region\":\"eight_decode_case_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),join(times,","),ok ? "true" : "false",ma,mr,path,length(out))
  ok||error("Fullattentionoutputoraclefailed")
 end
 println("PASS")
end

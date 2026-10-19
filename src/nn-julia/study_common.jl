using CUDA,Printf
include(joinpath(@__DIR__,"study_cases.jl"))
struct LatLong
 lat::Float32
 lng::Float32
end
function read_exact(path,T,count)
 bytes=read(path);length(bytes)==count*sizeof(T)||error("invalid asset bytes $path")
 collect(reinterpret(T,bytes))
end
function corrected_selection!(dist,ids,k)
 for i in 1:k
  minloc=i
  for j in i:length(dist)
   if dist[j]<dist[minloc];minloc=j;end
  end
  dist[i],dist[minloc]=dist[minloc],dist[i]
  ids[i],ids[minloc]=ids[minloc],ids[i]
 end
end
function nn_batch!(launch,payload,output,c)
 for _ in 1:c.repeats;launch(payload,output,c.n,c.lat,c.lng);end
 CUDA.device_synchronize(blocking=true)
end
function nn_study(launch,echo)
 input=read_exact(joinpath(@__DIR__,"input_f32.bin"),Float32,INPUT_FLOATS)
 oracle=read_exact(joinpath(@__DIR__,"oracle_f64.bin"),Float64,OUTPUT_COUNT)
 exact=read_exact(joinpath(@__DIR__,"oracle_exact_u8.bin"),UInt8,OUTPUT_COUNT)
 works=map(CASES)do c
  h=collect(reinterpret(LatLong,input[c.offset+1:c.offset+2*(Int(c.n)+16)]))
  d=CuArray(h);n=Int(c.n);reset=fill(91.25f0,n+32);reset[17:n+16].=NaN32
  o=CuArray(reset);e=CUDA.zeros(Float32,3)
  (c=c,d=d,output=o,payload=view(d,9:n+8),outpayload=view(o,17:n+16),reset=reset,echo=e)
 end
 # Warm the exact timed host loop, not just an isolated device specialization.
 for x in works
  copyto!(x.output,x.reset);nn_batch!(launch,x.payload,x.outpayload,x.c)
 end
 for trial in 1:5
  prefix=Float64[];outputs=Float64[];components=Float64[]
  for x in works
   c=x.c;n=Int(c.n);copyto!(x.output,x.reset);before=Array(x.d)
   echo(x.echo,c.n,c.lat,c.lng);echoed=Array(x.echo);CUDA.device_synchronize(blocking=true)
   start=time_ns();nn_batch!(launch,x.payload,x.outpayload,c);push!(components,(time_ns()-start)*1e-9/c.repeats)
   after=Array(x.d);raw=Array(x.output)
   append!(prefix,Float64[c.n,c.k,c.repeats,reinterpret(UInt32,c.lat),reinterpret(UInt32,c.lng),reinterpret(UInt32,echoed[1]),reinterpret(UInt32,echoed[2]),reinterpret(UInt32,echoed[3])])
   append!(prefix,Float64.(reinterpret(UInt32,before)));append!(prefix,Float64.(reinterpret(UInt32,after)))
   dist=raw[17:n+16];append!(outputs,Float64.(dist));ids=Int32.(0:n-1)
   corrected_selection!(dist,ids,c.k);append!(outputs,Float64.(dist));append!(outputs,Float64.(ids))
   append!(outputs,Float64.(reinterpret(UInt32,raw[1:16])));append!(outputs,Float64.(reinterpret(UInt32,raw[n+17:n+32])))
  end
  length(prefix)==PREFIX_COUNT||error("prefix count mismatch");append!(prefix,outputs);length(prefix)==OUTPUT_COUNT||error("artifact count mismatch")
  ok=true;ma=0.0;mr=0.0
  for i in eachindex(prefix)
   a=abs(prefix[i]-oracle[i]);ma=max(ma,a);mr=max(mr,a/max(abs(oracle[i]),1e-300))
   ok &= isfinite(prefix[i]) && (exact[i]==1 ? a==0 : a<=1e-5+8*Float64(eps(Float32))*abs(oracle[i]))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"nn_trial_$(trial).bin");write(path,prefix)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum5_original_distance_kernel_batch_means_R10000\",\"jit_excluded\":true,\"internal_repeats\":10000,\"component_time_seconds\":[",trial,sum(components),ok ? "true" : "false",length(prefix),ma,mr)
  for i in eachindex(components);i>1&&print(',');@printf("%.17g",components[i]);end
  @printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",path,length(prefix));flush(stdout)
  ok||error("full independent CPU check failed")
 end
end

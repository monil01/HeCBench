using Printf
include(joinpath(@__DIR__,"study_cases.jl"))
const G=32
const GUARD=UInt32(0xa5a5a5a5)
function study(allocate,launch)
 f=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"input_u32.bin"))))
 oracle=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin"))))
 states=[]
 for c in CASES
  sizes=[80*c.n,72,80*c.n];off=c.io;host=Vector{UInt32}[]
  for k in 1:3
   h=fill(GUARD,sizes[k]+2G)
   if k<3
    copyto!(h,G+1,f,off+1,sizes[k]);off+=sizes[k]
   end
   push!(host,h)
  end
  raw=[allocate(h) for h in host]
  dev=[view(a,G+1:length(a)-G) for a in raw]
  push!(states,(c=c,host=host,raw=raw,dev=dev))
 end
 function capture!(payload,s)
  for a in s.raw;append!(payload,Array(a));end
 end
 function invoke(s)
  launch(s.dev[1],s.dev[2],s.dev[3],Int32(s.c.n),s.c.threads)
 end
 function cchecked(v,s)
  c=s.c
  all(==(GUARD),view(v,1:G))&&all(==(GUARD),view(v,length(v)-G+1:length(v)))&&view(v,G+1:length(v)-G)==view(oracle,c.oo+1:c.oo+80*c.n)
 end
 for s in states
  invoke(s);CUDA.device_synchronize()
  cchecked(Array(s.raw[3]),s)||error("Warm full site AoS oracle")
 end
 for trial in 1:5
  pre=UInt32[];post=UInt32[];seconds=0.;ok=true;times=Float64[]
  for s in states
   copyto!(s.raw[3],s.host[3]);CUDA.device_synchronize();capture!(pre,s)
   start=time_ns()
   for r in 1:3;invoke(s);end
   CUDA.device_synchronize();tm=(time_ns()-start)*1e-9/3;seconds+=tm;push!(times,tm)
   for k in 1:3
    v=Array(s.raw[k]);ok&=k<3 ? v==s.host[k] : cchecked(v,s);append!(post,v)
   end
  end
  append!(pre,post)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"su3_trial$(trial).bin")
  open(io->write(io,pre),path,"w")
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":3,\"timing_region\":\"sum_seven_original_AoS_complex_matrix_batches_three_calls_completion\",\"max_abs_error\":null,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"},\"case_times_seconds\":[%s]}\n",trial,ok ? "true" : "false",seconds,path,length(pre),length(pre),join(times,','))
  ok||error("Full complex products, all AoS metadata, inputs and guards")
 end
end

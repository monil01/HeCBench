using Printf,SHA
const G=16;const GUARD=reinterpret(Int32,UInt32(0xa5a55a5a));const POISON=reinterpret(Int32,UInt32(0x13579bdf));const R=20
function run_study(transfer!)
 patterns=collect(reinterpret(Int32,read(joinpath(@__DIR__,"pattern_u32.bin"))))
 cases=Any[];prefix=Int32[]
 for mode in 0:1, sz in [1<<(i+6) for i in 0:15]
  n=sz÷4;payload=mode==0 ? fill(Int32(1),n) : patterns[1:n]
  host=vcat(fill(GUARD,G),payload,fill(GUARD,G));dev=CuArray(vcat(fill(GUARD,G),fill(POISON,n),fill(GUARD,G)))
  initial=Array(dev);poisoned=copy(initial);append!(prefix,host);append!(prefix,initial);append!(prefix,poisoned)
  push!(cases,(host=host,dev=dev,payload=payload,poisoned=poisoned,n=n,mode=mode,sz=sz))
 end
 # Warm all APIs/shapes outside every timing window; keep array owners alive through completion.
 for a in cases
  GC.@preserve a begin
   transfer!(view(a.dev,G+1:G+a.n),view(a.host,G+1:G+a.n));CUDA.device_synchronize()
   transfer!(view(a.host,G+1:G+a.n),view(a.dev,G+1:G+a.n));CUDA.device_synchronize()
  end
 end
 for trial in 1:5
  outputs=Int32[];seconds=Float64[];correct=true
  for a in cases
   copyto!(a.host,G+1,a.payload,1,a.n);copyto!(a.dev,a.poisoned);CUDA.device_synchronize()
   GC.@preserve a begin
    start=time_ns()
    for _ in 1:R;transfer!(view(a.dev,G+1:G+a.n),view(a.host,G+1:G+a.n));end
    CUDA.device_synchronize();push!(seconds,(time_ns()-start)*1e-9/R)
    h2d=Array(a.dev);correct &= h2d==vcat(fill(GUARD,G),a.payload,fill(GUARD,G)) && a.host==h2d;append!(outputs,h2d)
    # Distinct poisoned host payload before the checked D2H timed batch.
    copyto!(a.host,a.poisoned);CUDA.device_synchronize();start=time_ns()
    for _ in 1:R;transfer!(view(a.host,G+1:G+a.n),view(a.dev,G+1:G+a.n));end
    CUDA.device_synchronize();push!(seconds,(time_ns()-start)*1e-9/R)
    correct &= a.host==h2d;append!(outputs,a.host);again=Array(a.dev);correct &= again==h2d;append!(outputs,again)
   end
  end
  file=tempname()*".bin";open(file,"w") do io;write(io,vcat(prefix,outputs));end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":null,\"max_rel_error\":null,\"exact_word_mismatches\":%d,\"jit_excluded\":true,\"timing_region\":\"sum_32_cases_two_pageable_transfer_batches_plus_device_completion_per_repeat\",\"internal_repeats\":20,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,sum(seconds),string(correct),correct ? 0 : -1,join(seconds,","),file,length(prefix)+length(outputs),length(prefix)+length(outputs))
 end
end

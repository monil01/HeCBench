using CUDA,Printf
include("study_device.jl")
const NS=(7,8,11,12,15)
function langford_study(factory)
 expected=reinterpret(Int64,read(joinpath(@__DIR__,"expected_i64.bin")));buffers=[CUDA.zeros(Int64,1+4096*n) for n in NS];launches=[factory(buffers[j],Val(NS[j])) for j in eachindex(NS)]
 for f in launches;f();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  ts=Float64[]
  for j in eachindex(NS)
   fill!(buffers[j],0);CUDA.device_synchronize(blocking=true);t=time_ns();launches[j]();CUDA.device_synchronize(blocking=true);push!(ts,(time_ns()-t)*1e-9)
  end
  out=Int64[]
  for (n,b) in zip(NS,buffers)
   v=Array(b);count=v[1];0<=count<4096||error("capacityoverflow");rows=[Tuple(v[2+(i-1)*n:1+i*n]) for i in 1:count];sort!(rows);append!(out,[n,count,length(unique(rows))]);for row in rows;append!(out,row);end;append!(out,zeros(Int64,(4096-count)*n))
  end
  ok=out==expected;allok&=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"langford_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%d,\"internal_repeats\":1,\"jit_excluded\":true,\"timing_region\":\"sum_five_sourcecase_GPU_DFS_launch_completion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(ts),ts...,string(ok),length(out),ok ? 0 : 1,ok ? 0 : 1,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end

using CUDA,Printf
function constant_study(factory)
 base=CUDA.fill(Int32(1515870810),33);out=view(base,17:17);prepared=[factory(out,Val(w)) for  w in (1,2,4)];allok=true
 for trial in 1:5
  prefix=Int32[];outputs=Int32[];guards=Int32[];times=zeros(3);ok=true
  for fixture in 1:2
   data=fixture==1 ? ones(Int32,1024) : Int32[mod(j*17,23)-11 for j in 0:1023];expected=sum(Int64.(data))
   for (wi,w) in  enumerate((1,2,4))
    symbol,run=prepared[wi];fill!(base,Int32(1515870810));CUDA.device_synchronize(blocking=true);symbol[]=Tuple(data);run();CUDA.device_synchronize(blocking=true)
    if fixture==1
     start=time_ns();for _ in 1:10;run();end;CUDA.device_synchronize(blocking=true);times[wi]=(time_ns()-start)*1e-9/10
    end
    actual=collect(symbol[]);raw=Array(base);ok&=actual==data&&raw[17]==expected;append!(prefix,actual);append!(prefix,Int32[w,1024,4194304,10,fixture]);push!(outputs,raw[17])
    for j in 1:16;push!(guards,raw[j],raw[j+17]);ok&=raw[j]==1515870810&&raw[j+17]==1515870810;end
   end
  end
  allok&=ok;artifact=vcat(prefix,outputs,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":198,\"max_abs_error\":%d,\"max_rel_error\":null,\"timing_region\":\"sum_of_three_original_constant_width_launch_loop_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_cases\":3,\"untimed_controls\":3,\"component_time_seconds\":{\"width4\":%.12g,\"width8\":%.12g,\"width16\":%.12g},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ok ? 0 : 1,times[1],times[2],times[3],path,length(artifact))
 end
 allok||error("Constantmemorysymbol/input/scalaroutput/guards mismatch");println("PASS")
end

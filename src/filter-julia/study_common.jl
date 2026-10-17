using CUDA
using Printf
const STUDY_N=262144
function study_loop!(launch!,mode,count,repeats)
    for _ in 1:repeats
        CUDA.fill!(count,Int32(0));launch!(mode)
    end
end
function study_main(prepare,args=ARGS)
    n=STUDY_N;repeats=isempty(args) ? 20 : parse(Int,args[1]);repeats>0||error("positive repeats")
    input=Int32[Int32((i*65537+17)%n)-Int32(n÷2) for i in 0:n-1];expected=sort(filter(>(Int32(0)),input));src=CuArray(input);dst=CUDA.zeros(Int32,n);count=CUDA.zeros(Int32,1);launch! = prepare(dst,count,src,n)
    study_loop!(launch!,0,count,1);study_loop!(launch!,1,count,1);CUDA.synchronize();allok=true
    for trial in 1:5
        times=Float64[];output=Int32[];ok=true;gct=0.0;gcp=0
        for mode in (0,1)
            CUDA.fill!(dst,Int32(0));CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,mode,count,repeats);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
            k=Int(Array(count)[1]);k==length(expected)||error("Incorrect count $k");values=sort(Array(dst)[1:k]);ok &=values==expected;push!(output,Int32(k));append!(output,values)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"filter_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_shared_and_global_compaction_including_counter_reset_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"mode_times_seconds\":[%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),repeats,times[1],times[2],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end

using CUDA
using Printf
function study_sort!(launch!,exponent)
    for step in 0:exponent-1,stage in step:-1:0
        launch!(Int32(1<<(stage+1)),Int32(1<<(step-stage)))
    end
end
function study_main(prepare,args=ARGS)
    exponent=14;n=1<<exponent;repeat=isempty(args) ? 3 : parse(Int,args[1]);repeat>0||error("positive repeat");ccall(:srand,Cvoid,(Cuint,),Cuint(123));input=Int32[Int32(ccall(:rand,Cint,())%1000) for _ in 1:n];expected=sort(input)
    data=CuArray(input);launch! = prepare(data,n);study_sort!(launch!,exponent);CUDA.synchronize();allok=true
    for trial in 1:5
        elapsed=0.0;gct=0.0;gcp=0
        for rep in 1:repeat
            copyto!(data,input);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_sort!(launch!,exponent);CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
        end
        got=Array(data);err=maximum(abs.(Int64.(got).-Int64.(expected)));ok=got==expected;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bitonic_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"complete_14level_105stage_bitonic_sort_reset_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,err,repeat,gct,gcp,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end

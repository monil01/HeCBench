using CUDA
using Printf
function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    n=65536;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");ccall(:srand,Cvoid,(Cuint,),Cuint(512));input=UInt32[UInt32(ccall(:rand,Cint,())%16) for _ in 1:n];expected=similar(input)
    for i in 1:512:n
        expected[i:i+511]=sort(input[i:i+511])
    end
    src=CuArray(input);dst=CUDA.zeros(UInt32,n);launch! = prepare(src,dst,n);study_loop!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.fill!(dst,UInt32(0));CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num()
        got=Array(dst);ma=maximum(abs.(Int64.(got).-Int64.(expected)));ok=got==expected;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"split_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"block512_sort_kernel_only_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,ma,repeat,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end

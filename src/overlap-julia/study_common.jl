using CUDA
using Printf
function study_pipeline!(launch!,din,dout,hin,hout,streams,used,repeats)
    current=1;n=length(din[1])
    GC.@preserve din dout hin hout begin
        for _ in 1:repeats
            next=current%used+1
            CUDA.stream!(streams[current]) do
                launch!(current)
            end
            unsafe_copyto!(pointer(din[next]),pointer(hin[next]),n;async=true,stream=streams[next])
            unsafe_copyto!(pointer(hout[current]),pointer(dout[current]),n;async=true,stream=streams[current])
            current=next
        end
        CUDA.synchronize()
    end
end
function study_main(prepare,args=ARGS)
    n=262144;repeats=12;inner=Int32(5);hin=[CUDA.pin(Int32[Int32((i*13+s*7)%257)-Int32(128) for i in 0:n-1]) for s in 1:4];hout=[CUDA.pin(zeros(Int32,n)) for _ in 1:4];din=[CuArray(a) for a in hin];dout=[CUDA.zeros(Int32,n) for _ in 1:4];streams=[CUDA.CuStream() for _ in 1:4];launch! = prepare(din,dout,n,inner)
    study_pipeline!(launch!,din,dout,hin,hout,streams,4,repeats);CUDA.synchronize();allok=true
    for trial in 1:5
        output=Int32[];times=Float64[];ok=true;ma=0;gct=0.0;gcp=0
        for used in (1,4)
            for s in 1:4
                copyto!(din[s],hin[s]);CUDA.fill!(dout[s],Int32(0));fill!(hout[s],Int32(0))
            end
            CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_pipeline!(launch!,din,dout,hin,hout,streams,used,repeats);elapsed=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
            for s in 1:4
                expected=s<=used ? hin[s].+inner : zeros(Int32,n);e=maximum(abs.(hout[s].-expected));ma=max(ma,Int(e));ok &=e==0;append!(output,hout[s])
            end
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"overlap_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum_one_and_four_stream_pinned_H2D_kernel_D2H_pipeline_per_frame\",\"jit_excluded\":true,\"internal_repeats\":%d,\"stream_mode_times_seconds\":[%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeats,times[1],times[2],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end

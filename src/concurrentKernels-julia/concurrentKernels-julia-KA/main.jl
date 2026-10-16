using CUDA,KernelAbstractions
@kernel function clock_block_kernel!(output,slot,count)
    value=Int64(0)
    for i in Int64(0):count-Int64(1)
        value+=i%Int64(3)
    end
    @inbounds output[slot]=value
end
@kernel function sum_kernel!(data,n)
    t=@index(Local,Linear)
    scratch=@localmem Int64 (32,)
    value=Int64(0);i=t
    while i<=n
        @inbounds value+=data[i];i+=32
    end
    @inbounds scratch[t]=value
    @synchronize
    offset=16
    while offset>0
        if t<=offset
            @inbounds scratch[t]+=scratch[t+offset]
        end
        @synchronize
        offset÷=2
    end
    if t==1
        @inbounds data[1]=scratch[1]
    end
end
host_clock_sum(n)=div(n,Int64(3))*Int64(3)+(n%Int64(3)==Int64(2) ? Int64(1) : Int64(0))
function launch_producer!(data,slot,count,stream)
    CUDA.stream!(stream) do
        clock_block_kernel!(CUDA.CUDABackend(),1)(data,Int32(slot),count;ndrange=1)
    end
end
function launch_sum!(data,n,stream)
    CUDA.stream!(stream) do
        sum_kernel!(CUDA.CUDABackend(),32)(data,Int32(n);ndrange=32)
    end
end
function run_graph!(device,n,count,streams,events)
    for slot in 1:n
        launch_producer!(device,slot,count,streams[slot])
        CUDA.record(events[slot],streams[slot])
        CUDA.wait(events[slot],streams[end])
    end
    launch_sum!(device,n,streams[end])
    CUDA.synchronize(streams[end])
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4
    count=length(args)>=2 ? parse(Int64,args[2]) : Int64(120000)
    1<=n<=32||error("Study configuration requires1to32producer kernels")
    data=CUDA.zeros(Int64,32)
    # Convertonce, keephostownerlive: avoidwhole-array streamtracking serializingdisjointwrites.
    device=CUDA.cudaconvert(data)
    streams=[CUDA.CuStream() for _ in 1:n+1]
    events=[CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ in 1:n]
    CUDA.synchronize()
    GC.@preserve data begin
        run_graph!(device,n,count,streams,events)
        for trial in 1:5
            fill!(data,Int64(0));CUDA.synchronize()
            start=time_ns();run_graph!(device,n,count,streams,events);seconds=(time_ns()-start)*1e-9
            result=Array(data)[1:n];expected=fill(host_clock_sum(count),n);expected[1]*=n
            correct=result==expected
            artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
            open(artifact,"w") do io;write(io,result);end
            println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
             ",\"checked_values\":",n,",\"max_abs_error\":0.0,\"timing_region\":\"producer_streams_event_join_reduction_fence_no_d2h\",\"jit_excluded\":true,\"internal_repeats\":1,",
             "\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"int64\",\"count\":",n,",\"byte_order\":\"little\"}}")
            correct||return 1
        end
    end
    return 0
end
exit(main(ARGS))

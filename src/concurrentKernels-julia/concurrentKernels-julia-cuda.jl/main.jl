using CUDA
using Printf

function clock_block_kernel!(out, slot::Int32, clock_count::Int64)
    clock_offset = Int64(0)
    for i in Int64(0):(clock_count - Int64(1))
        clock_offset += i % Int64(3)
    end
    @inbounds out[slot] = clock_offset
    return
end

function sum_kernel!(clocks, n::Int32)
    shared = CuStaticSharedArray(Int64, 32)
    tid = threadIdx().x
    lane = tid - Int32(1)
    acc = Int64(0)
    i = lane
    while i < n
        @inbounds acc += clocks[i + Int32(1)]
        i += blockDim().x
    end
    @inbounds shared[tid] = acc
    sync_threads()

    offset = Int32(16)
    while offset > 0
        if lane < offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end

    if lane == 0
        @inbounds clocks[1] = shared[1]
    end
    return
end

function host_clock_sum(clock_count::Int64)
    groups = clock_count ÷ Int64(3)
    rem = clock_count % Int64(3)
    total = groups * Int64(3)
    if rem == Int64(2)
        total += Int64(1)
    end
    return total
end

function launch_producer!(data,slot,count,stream)
    @cuda threads=1 blocks=1 stream=stream clock_block_kernel!(data,Int32(slot),count)
end
function launch_sum!(data,n,stream)
    @cuda threads=32 blocks=1 stream=stream sum_kernel!(data,Int32(n))
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

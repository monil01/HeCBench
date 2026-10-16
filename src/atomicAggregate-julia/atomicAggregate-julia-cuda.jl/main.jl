using CUDA
using Printf

const CUDACore = CUDA.CUDACore

@inline function atomic_agg_inc!(d, idx::Int32)
    mask = UInt32(0)
    full_mask = UInt32(0xffffffff)
    lane0 = (threadIdx().x - Int32(1)) % Int32(32)

    for i in Int32(0):Int32(31)
        peer_idx = CUDACore.shfl_sync(full_mask, idx, i + Int32(1))
        peer_mask = CUDACore.vote_ballot_sync(full_mask, peer_idx == idx)
        if i == lane0
            mask = peer_mask
        end
    end

    leader0 = CUDACore.ffs(mask) - Int32(1)
    previous = Int32(0)

    if lane0 == leader0
        previous = CUDA.atomic_add!(pointer(d, idx), CUDACore.popc(mask))
    end

    previous = CUDACore.shfl_sync(mask, previous, leader0 + Int32(1))
    prior_lanes = mask & ((UInt32(1) << UInt32(lane0)) - UInt32(1))
    return previous + CUDACore.popc(prior_lanes)
end

function aggregate_kernel!(d, s::Int32)
    lane0 = threadIdx().x - Int32(1)
    idx = (lane0 % s) + Int32(1)  # CUDA's 0-based threadIdx.x % s, converted once to Julia indexing.
    atomic_agg_inc!(d, idx)
    return
end

function main(args)
    blocks=length(args)>=1 ? parse(Int,args[1]) : 64
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    sizes=(32,16,8,4,2,1)
    buffers=[CUDA.zeros(Int32,s) for s in sizes]
    allkeys=[CuArray(Int32[mod(t-1,s)+1 for t in 1:256]) for s in sizes]
    for (index,s) in enumerate(sizes)
        data=buffers[index];keys=allkeys[index]
        @cuda threads=256 blocks=blocks aggregate_kernel!(data,Int32(s))
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Int32[];correct=true
        for (index,s) in enumerate(sizes)
            data=buffers[index];keys=allkeys[index];fill!(data,Int32(0));CUDA.synchronize()
            start=time_ns()
            for _ in 1:repeat
                @cuda threads=256 blocks=blocks aggregate_kernel!(data,Int32(s))
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data);expected=Int32(div(256,s)*blocks*repeat)
            correct &= all(result.==expected);append!(values,result)
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":63,\"max_abs_error\":0.0,\"timing_region\":\"sum_six_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"int32\",\"count\":63,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

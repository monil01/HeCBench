using CUDA,KernelAbstractions
import Atomix
# Portable algorithm adaptation: shared-memory32-peer match scan replaces warpcollectives.
# Only leader updates each same-key group. Native helper return value is discarded.
@kernel function aggregate_kernel!(data,s,inputkeys)
    t=@index(Local,Linear)
    sharedkeys=@localmem Int32 (256,)
    @inbounds sharedkeys[t]=inputkeys[t]
    @synchronize
    @inbounds key=sharedkeys[t]
    base=div(t-1,32)*32
    leader=t;count=Int32(0)
    for lane in 1:32
        @inbounds if sharedkeys[base+lane]==key
            count+=Int32(1)
            leader=min(leader,base+lane)
        end
    end
    if t==leader
        Atomix.@atomic data[key]+=count
    end
end
function main(args)
    blocks=length(args)>=1 ? parse(Int,args[1]) : 64
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    sizes=(32,16,8,4,2,1)
    buffers=[CUDA.zeros(Int32,s) for s in sizes]
    allkeys=[CuArray(Int32[mod(t-1,s)+1 for t in 1:256]) for s in sizes]
    for (index,s) in enumerate(sizes)
        data=buffers[index];keys=allkeys[index]
        aggregate_kernel!(get_backend(data),256)(data,Int32(s),keys;ndrange=blocks*256)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Int32[];correct=true
        for (index,s) in enumerate(sizes)
            data=buffers[index];keys=allkeys[index];fill!(data,Int32(0));CUDA.synchronize()
            start=time_ns()
            for _ in 1:repeat
                aggregate_kernel!(get_backend(data),256)(data,Int32(s),keys;ndrange=blocks*256)
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

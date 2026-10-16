using CUDA
using Printf
using Random

function atomic_reduction!(input, output, n::Int32, width::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x * width
    start = tid0 * width
    local_sum = Int32(0)
    i = start
    while i < n
        for lane in Int32(0):(width - Int32(1))
            idx = i + lane
            if idx < n
                @inbounds local_sum += input[idx + Int32(1)]
            end
        end
        i += stride
    end
    CUDA.@atomic output[1] += local_sum
    return
end

function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 65536
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    input=Int32[mod(i-1,2) for i in 1:n];dx=CuArray(input);dy=CUDA.zeros(Int32,1)
    expected=Int32(sum(input));cases=[(bs,w) for bs in (128,256,512,1024) for w in (1,2,4,8,16)]
    for (bs,width) in cases
        blocks=max(div(min(cld(n,bs),2048),width),1)
        @cuda threads=bs blocks=blocks atomic_reduction!(dx,dy,Int32(n),Int32(width))
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Int32[];correct=true
        for (bs,width) in cases
            blocks=max(div(min(cld(n,bs),2048),width),1)
                CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                fill!(dy,Int32(0))
                @cuda threads=bs blocks=blocks atomic_reduction!(dx,dy,Int32(n),Int32(width))
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            actual=Array(dy)[1];push!(values,actual);correct &= actual==expected
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":20,\"max_abs_error\":0.0,\"timing_region\":\"sum_twenty_reset_plus_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"int32\",\"count\":20,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

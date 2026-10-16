using CUDA,KernelAbstractions
import Atomix
@kernel function atomic_reduction!(input,output,n,width,total_threads)
    tid=@index(Global,Linear)
    i=(tid-1)*width
    subtotal=Int32(0)
    while i<n
        for lane in Int32(0):width-Int32(1)
            idx=i+lane
            if idx<n
                @inbounds subtotal+=input[idx+1]
            end
        end
        i+=total_threads*width
    end
    Atomix.@atomic output[1]+=subtotal
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 65536
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    input=Int32[mod(i-1,2) for i in 1:n];dx=CuArray(input);dy=CUDA.zeros(Int32,1)
    expected=Int32(sum(input));cases=[(bs,w) for bs in (128,256,512,1024) for w in (1,2,4,8,16)]
    for (bs,width) in cases
        blocks=max(div(min(cld(n,bs),2048),width),1)
        atomic_reduction!(get_backend(dy),bs)(dx,dy,Int32(n),Int32(width),Int32(blocks*bs);ndrange=blocks*bs)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Int32[];correct=true
        for (bs,width) in cases
            blocks=max(div(min(cld(n,bs),2048),width),1)
                CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                fill!(dy,Int32(0))
                atomic_reduction!(get_backend(dy),bs)(dx,dy,Int32(n),Int32(width),Int32(blocks*bs);ndrange=blocks*bs)
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

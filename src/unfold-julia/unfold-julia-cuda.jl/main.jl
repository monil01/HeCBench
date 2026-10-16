using CUDA
using Printf
using Random

function unfold_backward!(grad_out, grad_in, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds grad_out[i] += grad_in[i]
        i += stride
    end
    return
end

function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 100
    input=Int32[mod(i-1,256) for i in 1:n]
    output=zeros(Int32,n);dx=CuArray(input);dy=CuArray(output)
    @cuda threads=256 blocks=min(cld(n,256),4096) unfold_backward!(dy,dx,Int32(n))
    CUDA.synchronize()
    for trial in 1:5
        fill!(dy,Int32(0));CUDA.synchronize()
        start=time_ns()
        for _ in 1:repeat
            @cuda threads=256 blocks=min(cld(n,256),4096) unfold_backward!(dy,dx,Int32(n))
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
        result=Array(dy);correct=result==Int32(repeat).*input
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
          ",\"checked_values\":",n,",\"max_abs_error\":",maximum(abs.(result.-Int32(repeat).*input)),
          ",\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
          ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"int32\",\"count\":",n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

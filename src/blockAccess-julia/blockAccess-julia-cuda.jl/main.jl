using CUDA
using Printf
using Random

function float_to_uchar(x::Float32)
    v = unsafe_trunc(Int32, x)
    return UInt8(v & Int32(0xff))
end

function reference_kernel!(out, input, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds out[idx] = float_to_uchar(input[idx])
        idx += stride
    end
    return
end

function blockaccess_kernel!(out, input, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds out[idx] = float_to_uchar(input[idx])
        idx += stride
    end
    return
end

function launch!(out,input,n,blocks,phase)
    if phase==1
        @cuda threads=256 blocks=blocks reference_kernel!(out,input,Int32(n))
    else
        @cuda threads=256 blocks=blocks blockaccess_kernel!(out,input,Int32(n))
    end
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 65536
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(n,1024);input=Float32[Float32(mod((i-1)*37,1024)-512)/4f0 for i in 1:n]
    expected=float_to_uchar.(input);dx=CuArray(input);dy=CUDA.zeros(UInt8,n)
    for phase in 1:2
        launch!(dy,dx,n,blocks,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=UInt32[];correct=true
        for phase in 1:2
            fill!(dy,UInt8(0));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                launch!(dy,dx,n,blocks,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(dy);correct &= result==expected;append!(values,UInt32.(result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",2n,",\"max_abs_error\":0.0,\"timing_region\":\"sum_two_float_to_byte_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"uint32\",\"count\":",2n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

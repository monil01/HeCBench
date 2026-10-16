using CUDA
using Printf

function flip_kernel!(input, output, n::Int64, dim_size::Int64)
    idx0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if idx0 >= n
        return
    end
    i0 = idx0 ÷ (dim_size * dim_size)
    rem = idx0 - i0 * dim_size * dim_size
    i1 = rem ÷ dim_size
    i2 = rem - i1 * dim_size
    src = (dim_size - 1 - i0) * dim_size * dim_size +
          (dim_size - 1 - i1) * dim_size +
          (dim_size - 1 - i2)
    @inbounds output[idx0 + 1] = input[src + 1]
    return
end

function main(args)
    d=length(args)>=1 ? parse(Int,args[1]) : 16
    repeat=length(args)>=2 ? parse(Int,args[2]) : 100
    n=d^3;input=Float32.(0:n-1);dx=CuArray(input);dy=CUDA.zeros(Float32,n)
    expected=reverse(input)
    @cuda threads=256 blocks=cld(n,256) flip_kernel!(dx,dy,Int64(n),Int64(d))
    CUDA.synchronize()
    for trial in 1:5
        fill!(dy,0f0);CUDA.synchronize();start=time_ns()
        for _ in 1:repeat
            @cuda threads=256 blocks=cld(n,256) flip_kernel!(dx,dy,Int64(n),Int64(d))
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
        result=Array(dy);correct=result==expected
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
          ",\"checked_values\":",n,",\"max_abs_error\":",maximum(abs.(result.-expected)),
          ",\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
          ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

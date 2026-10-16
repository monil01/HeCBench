using CUDA,KernelAbstractions
# Reverse all three axes in a contiguous cubic tensor; row-major index mapping.
@kernel function flip_kernel!(input,output,n,d)
    index=@index(Global,Linear)
    if index<=n
        z=index-1
        i0=z÷(d*d); i1=(z÷d)%d; i2=z%d
        src=(d-1-i0)*d*d+(d-1-i1)*d+(d-1-i2)
        @inbounds output[index]=input[src+1]
    end
end
function main(args)
    d=length(args)>=1 ? parse(Int,args[1]) : 16
    repeat=length(args)>=2 ? parse(Int,args[2]) : 100
    n=d^3;input=Float32.(0:n-1);dx=CuArray(input);dy=CUDA.zeros(Float32,n)
    expected=reverse(input)
    flip_kernel!(get_backend(dy),256)(dx,dy,Int64(n),Int64(d);ndrange=n)
    CUDA.synchronize()
    for trial in 1:5
        fill!(dy,0f0);CUDA.synchronize();start=time_ns()
        for _ in 1:repeat
            flip_kernel!(get_backend(dy),256)(dx,dy,Int64(n),Int64(d);ndrange=n)
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

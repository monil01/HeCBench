using CUDA, KernelAbstractions
@kernel function saxpy_kernel!(y,x,a,n)
    i=@index(Global,Linear)
    if i<=n
        @inbounds y[i]=a*x[i]+y[i]
    end
end
function main()
    n = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 4096
    repeat = length(ARGS)>=2 ? parse(Int,ARGS[2]) : 100
    a=2.0f0
    # Exact zero-based CUDA initialization, unlike historical one-based Julia port.
    x=[1f0+Float32(((i-1)*2654435761)&0xffff)/65535f0 for i in 1:n]
    y=[0.5f0+Float32(((i-1)*40503)&0xffff)/65535f0 for i in 1:n]
    expected=y .+ Float32(repeat).*a.*x
    dx=CuArray(x); dy=CuArray(y)
    saxpy_kernel!(get_backend(dy),256)(dy,dx,a,Int32(n);ndrange=n)
    CUDA.synchronize()
    for trial in 1:5
        copyto!(dy,y); CUDA.synchronize()
        start=time_ns()
        for _ in 1:repeat
            saxpy_kernel!(get_backend(dy),256)(dy,dx,a,Int32(n);ndrange=n)
        end
        CUDA.synchronize()
        seconds=(time_ns()-start)*1e-9/repeat
        result=Array(dy)
        error=maximum(abs.(result.-expected))
        correct=all(isfinite,result)&&error<1f-2
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io; write(io,result); end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
          ",\"checked_values\":",n,",\"max_abs_error\":",error,",\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
          ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",n,",\"byte_order\":\"little\"}}")
        correct || return 1
    end
    return 0
end
exit(main())

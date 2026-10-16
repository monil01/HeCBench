using CUDA,KernelAbstractions
# Native configuration has idx_dim=0,size=2,step=1,strides=1: one contribution.
@kernel function unfold_backward!(output,input,n)
    i=@index(Global,Linear)
    if i<=n
        @inbounds output[i]+=input[i]
    end
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 100
    input=Int32[mod(i-1,256) for i in 1:n]
    output=zeros(Int32,n);dx=CuArray(input);dy=CuArray(output)
    unfold_backward!(get_backend(dy),256)(dy,dx,Int32(n);ndrange=n)
    CUDA.synchronize()
    for trial in 1:5
        fill!(dy,Int32(0));CUDA.synchronize()
        start=time_ns()
        for _ in 1:repeat
            unfold_backward!(get_backend(dy),256)(dy,dx,Int32(n);ndrange=n)
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

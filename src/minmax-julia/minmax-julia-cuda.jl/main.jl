using CUDA
make_state(n)=CUDA.zeros(Float32,n)
function evaluate!(magnitudes,dx,dy,x,y,n,phase)
    magnitudes.=dx.*dx.+dy.*dy
    imin=argmin(magnitudes);imax=argmax(magnitudes)
    return Float32[x[imin],y[imin],x[imax],y[imax]]
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    x=Float32[mod((i-1)*37,n) for i in 1:n]
    y=Float32[Float32(mod((i-1)*19,n))/Float32(n) for i in 1:n]
    dx=CuArray(x);dy=CuArray(y);state=make_state(n)
    magnitudes=x.*x.+y.*y;imin=argmin(magnitudes);imax=argmax(magnitudes)
    expected=Float32[x[imin],y[imin],x[imax],y[imax]]
    for phase in 1:2
        evaluate!(state,dx,dy,x,y,n,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float32[];correct=true
        for phase in 1:2
            CUDA.synchronize();start=time_ns();result=Float32[]
            for _ in 1:repeat
                result=evaluate!(state,dx,dy,x,y,n,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            correct &= result==expected;append!(values,result)
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":8,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_extrema_operations_host_results_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":8,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

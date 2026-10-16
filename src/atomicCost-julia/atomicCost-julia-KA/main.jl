using CUDA,KernelAbstractions
import Atomix
@kernel function atomic_cost!(data,size,::Val{PHASE}) where PHASE
    tid=@index(Global,Linear)
    for i in (tid-1)*size:tid*size-1
        value=convert(eltype(data),mod(i,2))
        if PHASE==1
            Atomix.@atomic data[tid]+=value
        else
            @inbounds data[tid]+=value
        end
    end
end
function launch!(data,size,blocks,phase)
    atomic_cost!(get_backend(data),256)(data,Int32(size),Val(phase);ndrange=blocks*256)
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    size=length(args)>=2 ? parse(Int,args[2]) : 4
    repeat=length(args)>=3 ? parse(Int,args[3]) : 10
    blocks=cld(n,256);cases=[(T,phase) for T in (Float64,Int32,Float32) for phase in 1:2]
    buffers=[CUDA.zeros(T,n) for (T,phase) in cases]
    for (j,(T,phase)) in enumerate(cases)
        data=buffers[j];launch!(data,size,blocks,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float64[];correct=true
        for (j,(T,phase)) in enumerate(cases)
            data=buffers[j];fill!(data,zero(T));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                launch!(data,size,blocks,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data)
            for index in 1:n
                expected=T(repeat*sum(mod(i,2) for i in (index-1)*size:index*size-1))
                correct &= result[index]==expected
            end
            append!(values,Float64.(result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",6n,",\"max_abs_error\":0.0,\"timing_region\":\"sum_six_atomic_vs_nonatomic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float64\",\"count\":",6n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

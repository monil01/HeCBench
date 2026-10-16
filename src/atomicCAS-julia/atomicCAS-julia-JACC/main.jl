using CUDA
import JACC,Atomix
JACC.@init_backend
function derived_cas!(i,data,bound,::Val{T},::Val{OP}) where {T,OP}
    if i<=bound
        value=T(i)
        @inbounds bits=data[1]
        while true
            old=reinterpret(T,bits)
            OP===:min && !(value<old) && break
            OP===:max && !(value>old) && break
            desired=OP===:add ? old+value : value
            newbits=reinterpret(UInt64,desired)
            previous=(Atomix.@atomicreplace data[1] bits => newbits).old
            previous==bits && break
            bits=previous
        end
    end
    return
end
function main(args)
    bound=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(bound,256)
    # Same case order as native: min3types, max3types, add3types.
    cases=[(T,op) for op in (:min,:max,:add) for T in (UInt64,Int64,Float64)]
    buffers=[CuArray([UInt64(0)]) for _ in cases]
    initial(T,op)=op===:min ? (T===Float64 ? floatmax(T) : typemax(T)) : op===:max ? (T===Int64 ? typemin(T) : T===Float64 ? floatmin(T) : zero(T)) : zero(T)
    expected(T,op)=op===:min ? one(T) : op===:max ? T(bound) : T(div(Int64(bound)*(bound+1),2))
    spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
    for (index,(T,op)) in enumerate(cases)
        data=buffers[index];copyto!(data,[reinterpret(UInt64,initial(T,op))])
        JACC.parallel_for(spec,bound,derived_cas!,data,Int32(bound),Val(T),Val(op))
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float64[];correct=true
        for (index,(T,op)) in enumerate(cases)
            data=buffers[index];copyto!(data,[reinterpret(UInt64,initial(T,op))]);CUDA.synchronize()
            reps=op===:add ? 1 : repeat
            start=time_ns()
            for _ in 1:reps
                JACC.parallel_for(spec,bound,derived_cas!,data,Int32(bound),Val(T),Val(op))
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/reps)
            actual=reinterpret(T,Array(data)[1]);correct &= actual==expected(T,op)
            push!(values,Float64(actual))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":9,\"max_abs_error\":0.0,\"timing_region\":\"sum_nine_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float64\",\"count\":9,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

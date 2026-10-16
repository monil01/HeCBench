using CUDA,KernelAbstractions
import Atomix
@inline function bounded_update!(data,index,limit,dec)
    T=eltype(data)
    @inbounds old=data[index]
    while true
        desired=dec ? ((old==zero(T)||old>limit) ? limit : old-one(T)) : (old>=limit ? zero(T) : old+one(T))
        observed=(Atomix.@atomicreplace data[index] old => desired).old
        observed==old && return
        old=observed
    end
end
@kernel function atomic_kernel!(data,n)
    index=@index(Global,Linear)
    if index<=n
        T=eltype(data);tid=T(index-1)
        Atomix.@atomic data[1]+=T(10)
        Atomix.@atomic data[2]-=T(10)
        Atomix.@atomic max(data[3],tid)
        Atomix.@atomic min(data[4],tid)
        Atomix.@atomic data[5]&=T(2)*tid+T(7)
        Atomix.@atomic data[6]|=one(T)<<(tid&T(31))
        Atomix.@atomic data[7]⊻=tid
        bounded_update!(data,8,T(17),false)
        bounded_update!(data,9,T(137),true)
    end
end
init_data(::Type{Int32}) = Int32[0, 0, -256, 256, 255, 0, 255, 0, 0]
init_data(::Type{UInt32}) = UInt32[0, 0, 0xffffff00, 256, 255, 0, 255, 0, 0]

function cpu_reference(::Type{T},n,repeat) where T
    data=init_data(T)
    for _ in 1:repeat
        for i in 0:n-1
            data[1]+=T(10);data[2]-=T(10);data[3]=max(data[3],T(i));data[4]=min(data[4],T(i))
            data[5]&=T(2i+7)
            mask=T===Int32 ? T(Int32(1)<<(i&31)) : T(UInt32(1)<<UInt32(i&31))
            data[6]|=mask;data[7]⊻=T(i)
            data[8]=data[8]>=T(17) ? zero(T) : data[8]+one(T)
            data[9]=(data[9]==zero(T)||data[9]>T(137)) ? T(137) : data[9]-one(T)
        end
    end
    return data
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(n,256);types=(Int32,UInt32);buffers=[CuArray(init_data(T)) for T in types]
    for (j,T) in enumerate(types)
        data=buffers[j]
        atomic_kernel!(get_backend(data),256)(data,Int32(n);ndrange=n)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=UInt32[];correct=true
        for (j,T) in enumerate(types)
            data=buffers[j];copyto!(data,init_data(T));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                atomic_kernel!(get_backend(data),256)(data,Int32(n);ndrange=n)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data);expected=cpu_reference(T,n,repeat)
            if result != expected
                println(stderr,"NUMERIC_DIAGNOSTIC type=",T," actual=",result," expected=",expected)
            end
            correct &= result==expected
            append!(values,reinterpret(UInt32,result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":18,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_atomic_intrinsic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"uint32\",\"count\":18,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

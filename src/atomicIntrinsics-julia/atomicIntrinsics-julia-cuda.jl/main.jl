using CUDA
@inline function bounded_update!(ptr,one,zero,limit,dec)
    old=CUDA.atomic_cas!(ptr,zero,zero)
    while true
        desired=dec ? ((old==zero||old>limit) ? limit : old-one) : (old>=limit ? zero : old+one)
        observed=CUDA.atomic_cas!(ptr,old,desired)
        observed==old && return
        old=observed
    end
end
atomic_inc_mod!(ptr,one,zero,limit)=bounded_update!(ptr,one,zero,limit,false)
atomic_dec_mod!(ptr,one,zero,limit)=bounded_update!(ptr,one,zero,limit,true)
function atomic_kernel!(data, len::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid >= len
        return
    end

    CUDA.atomic_add!(pointer(data, 1), Int32(10))
    CUDA.atomic_sub!(pointer(data, 2), Int32(10))
    CUDA.atomic_max!(pointer(data, 3), tid)
    CUDA.atomic_min!(pointer(data, 4), tid)
    CUDA.atomic_and!(pointer(data, 5), Int32(2) * tid + Int32(7))
    CUDA.atomic_or!(pointer(data, 6), Int32(1) << (tid & Int32(31)))
    CUDA.atomic_xor!(pointer(data, 7), tid)
    atomic_inc_mod!(pointer(data, 8), Int32(1), Int32(0), Int32(17))
    atomic_dec_mod!(pointer(data, 9), Int32(1), Int32(0), Int32(137))
    return
end

function atomic_kernel!(data::CuDeviceVector{UInt32, 1}, len::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid >= len
        return
    end

    utid = UInt32(tid)
    CUDA.atomic_add!(pointer(data, 1), UInt32(10))
    CUDA.atomic_sub!(pointer(data, 2), UInt32(10))
    CUDA.atomic_max!(pointer(data, 3), utid)
    CUDA.atomic_min!(pointer(data, 4), utid)
    CUDA.atomic_and!(pointer(data, 5), UInt32(2) * utid + UInt32(7))
    CUDA.atomic_or!(pointer(data, 6), UInt32(1) << (utid & UInt32(31)))
    CUDA.atomic_xor!(pointer(data, 7), utid)
    atomic_inc_mod!(pointer(data, 8), UInt32(1), UInt32(0), UInt32(17))
    atomic_dec_mod!(pointer(data, 9), UInt32(1), UInt32(0), UInt32(137))
    return
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
        @cuda threads=256 blocks=blocks atomic_kernel!(data,Int32(n))
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=UInt32[];correct=true
        for (j,T) in enumerate(types)
            data=buffers[j];copyto!(data,init_data(T));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                @cuda threads=256 blocks=blocks atomic_kernel!(data,Int32(n))
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data);correct &= result==cpu_reference(T,n,repeat)
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

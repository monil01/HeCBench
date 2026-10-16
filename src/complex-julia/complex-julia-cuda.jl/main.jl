# Exactnativeperthread LCG: UInt64wrap thenmod2^63, seed1 fastforwardzeroindex.
@inline function native_random4(index)
    n=UInt64(index);a=UInt64(2806196910506780709);c=UInt64(1);an=UInt64(1);cn=UInt64(0)
    while n>0
        if (n&UInt64(1))!=0
            an*=a;cn=cn*a+c
        end
        c*=a+UInt64(1);a*=a;n>>=1
    end
    seed=(an+cn)&UInt64(0x7fffffffffffffff)
    seed=(UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff);r1=Float64(seed)/9223372036854775808.0
    seed=(UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff);r2=Float64(seed)/9223372036854775808.0
    seed=(UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff);r3=Float64(seed)/9223372036854775808.0
    seed=(UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff);r4=Float64(seed)/9223372036854775808.0
    return r1,r2,r3,r4
end
using CUDA
using Printf

function complex_float_kernel!(check, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    r1,r2,r3,r4=native_random4(i-1)
    z1=ComplexF32(Float32(r1),Float32(r2))
    z2=ComplexF32(Float32(r3),Float32(r4))

    s = UInt8(0)
    s += UInt8(abs(abs(z1 * z2) - abs(z1) * abs(z2)) < 1.0f-3)
    s += UInt8(abs(abs(z1 + z2)^2 - real((z1 + z2) * (conj(z1) + conj(z2)))) < 1.0f-3)
    s += UInt8(abs(abs(z1 - z2)^2 - real((z1 - z2) * (conj(z1) - conj(z2)))) < 1.0f-3)
    s += UInt8(abs(real(z1 * conj(z2) + z2 * conj(z1)) -
                  2.0f0 * (real(z1) * real(z2) + imag(z1) * imag(z2))) < 1.0f-3)
    s += UInt8(abs(abs(conj(z1) / z2) - abs(conj(z1) / conj(z2))) < 1.0f-3)
    @inbounds check[i] = s
    return
end

function complex_double_kernel!(check, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    r1,r2,r3,r4=native_random4(i-1)
    z1=ComplexF64(Float64(r1),Float64(r2))
    z2=ComplexF64(Float64(r3),Float64(r4))

    s = UInt8(0)
    s += UInt8(abs(abs(z1 * z2) - abs(z1) * abs(z2)) < 1.0e-3)
    s += UInt8(abs(abs(z1 + z2)^2 - real((z1 + z2) * (conj(z1) + conj(z2)))) < 1.0e-3)
    s += UInt8(abs(abs(z1 - z2)^2 - real((z1 - z2) * (conj(z1) - conj(z2)))) < 1.0e-3)
    s += UInt8(abs(real(z1 * conj(z2) + z2 * conj(z1)) -
                  2.0 * (real(z1) * real(z2) + imag(z1) * imag(z2))) < 1.0e-3)
    s += UInt8(abs(abs(conj(z1) / z2) - abs(conj(z1) / conj(z2))) < 1.0e-3)
    @inbounds check[i] = s
    return
end

function launch!(data,n,blocks,phase)
    if phase<=2
        @cuda threads=256 blocks=blocks complex_float_kernel!(data,Int32(n))
    else
        @cuda threads=256 blocks=blocks complex_double_kernel!(data,Int32(n))
    end
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(n,256);data=CUDA.zeros(UInt8,n)
    for phase in 1:4
        launch!(data,n,blocks,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=UInt32[];correct=true
        for phase in 1:4
            fill!(data,UInt8(0));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                launch!(data,n,blocks,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data);correct &= all(result.==UInt8(5));append!(values,UInt32.(result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",4n,",\"max_abs_error\":0.0,\"timing_region\":\"sum_four_complex_property_score_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"uint32\",\"count\":",4n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

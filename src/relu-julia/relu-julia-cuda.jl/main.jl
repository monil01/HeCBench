using CUDA
using Printf
using Random

const THREADS = 256

function relugrad_kernel!(gradient, feature, backprop, count::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= count
        @inbounds backprop[i] = feature[i] > Float16(0) ? gradient[i] : Float16(0)
        i += stride
    end
    return
end

@inline function signed_byte(x::UInt32, shift::UInt32)
    b = Int32((x >> shift) & UInt32(0xff))
    return b >= Int32(128) ? b - Int32(256) : b
end

@inline function relu_pack(x::UInt32)
    b0 = max(signed_byte(x, UInt32(0)), Int32(0))
    b1 = max(signed_byte(x, UInt32(8)), Int32(0))
    b2 = max(signed_byte(x, UInt32(16)), Int32(0))
    b3 = max(signed_byte(x, UInt32(24)), Int32(0))
    return UInt32(b0) | (UInt32(b1) << UInt32(8)) |
           (UInt32(b2) << UInt32(16)) | (UInt32(b3) << UInt32(24))
end

function relu_kernel!(count::Int32, input, output)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= count
        @inbounds output[i] = relu_pack(input[i])
    end
    return
end

function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 100
    feature=Float16[Float32(mod(i-1,17)-8)/8f0 for i in 1:n]
    gradient=fill(Float16(1),n)
    packed=UInt32[UInt32(((i-1)*2654435761)&0xffffffff) for i in 1:n]
    expected_grad=[feature[i]>Float16(0) ? gradient[i] : Float16(0) for i in 1:n]
    expected_pack=relu_pack.(packed)
    dg=CuArray(gradient);df=CuArray(feature);db=CUDA.zeros(Float16,n)
    dx=CuArray(packed);dy=CUDA.zeros(UInt32,n)
    @cuda threads=256 blocks=cld(n,256) relugrad_kernel!(dg,df,db,Int32(n))
    @cuda threads=256 blocks=cld(n,256) relu_kernel!(Int32(n),dx,dy)
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];allvalues=Float64[];correct=true
        for phase in 1:4
            fill!(db,Float16(0));fill!(dy,UInt32(0));CUDA.synchronize()
            start=time_ns()
            if phase<=2
                for _ in 1:repeat
                    @cuda threads=256 blocks=cld(n,256) relugrad_kernel!(dg,df,db,Int32(n))
                end
            else
                for _ in 1:repeat
                    @cuda threads=256 blocks=cld(n,256) relu_kernel!(Int32(n),dx,dy)
                end
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            if phase<=2
                output=Array(db);correct &= output==expected_grad
            else
                output=Array(dy);correct &= output==expected_pack
            end
            append!(allvalues,Float64.(output))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,allvalues);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",4n,",\"max_abs_error\":0.0,\"timing_region\":\"sum_four_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float64\",\"count\":",4n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

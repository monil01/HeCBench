using CUDA
using Printf

function truncate_factor(factor::Float32, x::Float32)
    return (factor + x) - factor
end

function init_arrays!(x, total::Int64)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        raw = Float32((i * Int64(1103515245) + Int64(12345)) & Int64(0x00ffffff)) / 16777216.0f0
        sign = (i & Int64(1)) == 0 ? 1.0f0 : -1.0f0
        x[i + 1] = sign * raw
        i += stride
    end
    return
end

function sum_array!(factor::Float32, len::Int32, x, result)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < len
        q = truncate_factor(factor, x[i + Int32(1)])
        CUDA.@atomic result[1] += q
        i += stride
    end
    return
end

function sum_arrays!(narrays::Int32, len::Int32, x, result, factor::Float32)
    arr = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while arr < narrays
        base = Int64(arr) * Int64(len)
        s = 0.0f0
        i = len - Int32(1)
        while i >= 0
            s += truncate_factor(factor, x[base + Int64(i) + 1])
            i -= Int32(1)
        end
        result[arr + Int32(1)] = s
        arr += stride
    end
    return
end

function create_rounding_factor(maxval::Float32, n::Int)
    delta = (maxval * Float32(n)) / (1.0f0 - 2.0f0 * Float32(n) * eps(Float32))
    _, exp = frexp(delta)
    return ldexp(1.0f0, exp)
end

function launch!(output,input,arrays,len,factor,phase)
    if phase==1
        for a in 1:arrays
            x=view(input,(a-1)*len+1:a*len);y=view(output,a:a)
            @cuda threads=256 blocks=256 sum_array!(factor,Int32(len),x,y)
        end
    else
        @cuda threads=256 blocks=256 sum_arrays!(Int32(arrays),Int32(len),input,output,factor)
    end
end
function main(args)
    arrays=length(args)>=1 ? parse(Int,args[1]) : 16
    len=length(args)>=2 ? parse(Int,args[2]) : 512
    n=arrays*len;factor=create_rounding_factor(1f0,len)
    input=Float32[]
    for i in 0:n-1
        raw=Float32((Int64(i)*1103515245+12345)&0x00ffffff)/16777216f0
        push!(input,mod(i,len)==0 ? 1f0 : (iseven(i) ? raw : -raw))
    end
    expected=zeros(Float32,arrays)
    for array in 1:arrays
        for index in 1:len
            expected[array]+=truncate_factor(factor,input[(array-1)*len+index])
        end
    end
    dx=CuArray(input);dy=CUDA.zeros(Float32,arrays)
    for phase in 1:2
        fill!(dy,0f0);launch!(dy,dx,arrays,len,factor,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float32[];correct=true
        for phase in 1:2
            fill!(dy,0f0);CUDA.synchronize();start=time_ns()
            launch!(dy,dx,arrays,len,factor,phase)
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9)
            result=Array(dy);correct &= reinterpret(UInt32,result)==reinterpret(UInt32,expected);append!(values,result)
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",2arrays,",\"max_abs_error\":0.0,\"timing_region\":\"sum_atomic_per_array_launches_and_reverse_single_launch\",\"jit_excluded\":true,\"internal_repeats\":1,",
         "\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",2arrays,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

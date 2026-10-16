using CUDA,KernelAbstractions
import Atomix
truncate_factor(factor::Float32,x::Float32)=(factor+x)-factor
@kernel function sum_array!(input,output,len,factor,array,total_threads)
    tid=@index(Global,Linear);i=tid
    while i<=len
        @inbounds q=truncate_factor(factor,input[(array-1)*len+i])
        Atomix.@atomic output[array]+=q
        i+=total_threads
    end
end
@kernel function sum_arrays!(input,output,arrays,len,factor,total_threads)
    array=@index(Global,Linear)
    while array<=arrays
        value=0f0
        for i in len:-1:1
            @inbounds value+=truncate_factor(factor,input[(array-1)*len+i])
        end
        @inbounds output[array]=value
        array+=total_threads
    end
end
function launch!(output,input,arrays,len,factor,phase)
    backend=get_backend(output)
    if phase==1
        for a in 1:arrays
            sum_array!(backend,256)(input,output,Int32(len),factor,Int32(a),Int32(65536);ndrange=65536)
        end
    else
        sum_arrays!(backend,256)(input,output,Int32(arrays),Int32(len),factor,Int32(65536);ndrange=65536)
    end
end
function create_rounding_factor(maxval::Float32, n::Int)
    delta = (maxval * Float32(n)) / (1.0f0 - 2.0f0 * Float32(n) * eps(Float32))
    _, exp = frexp(delta)
    return ldexp(1.0f0, exp)
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

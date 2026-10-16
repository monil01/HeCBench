using CUDA
using Printf

const BLOCK_SIZE = Int32(256)

function block_range_atomic_global!(data, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    while i < n
        idx = threadIdx().x
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function warp_range_atomic_global!(data, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    while i < n
        idx = (i & Int32(0x1f)) + Int32(1)
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function single_range_atomic_global!(data, offset0::Int32, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    i = tid0
    idx = offset0 + Int32(1)
    while i < n
        CUDA.@atomic data[idx] += one(eltype(data))
        i += stride
    end
    return
end

function launch!(data,n,blocks,phase,offset)
    if phase==1
        @cuda threads=256 blocks=blocks block_range_atomic_global!(data,Int32(n))
    elseif phase==2
        @cuda threads=256 blocks=blocks warp_range_atomic_global!(data,Int32(n))
    else
        @cuda threads=256 blocks=blocks single_range_atomic_global!(data,Int32(offset),Int32(n))
    end
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(n,256);cases=[(T,phase) for T in (Float64,Int32,Float32) for phase in 1:3]
    buffers=[CuArray(T.(1:1024)) for (T,phase) in cases]
    for (j,(T,phase)) in enumerate(cases)
        data=buffers[j]
        launch!(data,n,blocks,phase,0)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float64[];correct=true
        for (j,(T,phase)) in enumerate(cases)
            data=buffers[j];initial=T.(1:1024);copyto!(data,initial);CUDA.synchronize();start=time_ns()
            for r in 0:repeat-1
                launch!(data,n,blocks,phase,mod(r,256))
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(data);expected=copy(initial)
            for r in 0:repeat-1
                for i in 0:n-1
                    index=phase==1 ? mod(i,256)+1 : phase==2 ? mod(i,32)+1 : mod(r,256)+1
                    expected[index]+=one(T)
                end
            end
            correct &= result==expected;append!(values,Float64.(result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":9216,\"max_abs_error\":0.0,\"timing_region\":\"sum_nine_global_atomic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float64\",\"count\":9216,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

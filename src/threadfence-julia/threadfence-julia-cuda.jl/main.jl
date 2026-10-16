using CUDA
using Printf

const BLOCKS = 256

function partial_sum_kernel!(array, result, n::Int32)
    shared = CuStaticSharedArray(Float32, BLOCKS)
    lid = threadIdx().x
    gid = (blockIdx().x - Int32(1)) * blockDim().x + lid

    value = Float32(0)
    if gid <= n
        value = array[gid]
    end
    shared[lid] = value
    sync_threads()

    stride = blockDim().x >>> Int32(1)
    while stride > 0
        if lid <= stride
            shared[lid] += shared[lid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    if lid == Int32(1)
        result[blockIdx().x] = shared[1]
    end
    return
end

function finalize_sum_kernel!(result, grids::Int32)
    shared = CuStaticSharedArray(Float32, BLOCKS)
    lid = threadIdx().x
    total = Float32(0)
    i = lid
    while i <= grids
        total += result[i]
        i += blockDim().x
    end
    shared[lid] = total
    sync_threads()

    stride = blockDim().x >>> Int32(1)
    while stride > 0
        if lid <= stride
            shared[lid] += shared[lid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    if lid == Int32(1)
        result[1] = shared[1]
    end
    return
end

function launch(input,output,counter,n,grids)
    @cuda threads=256 blocks=grids partial_sum_kernel!(input,output,Int32(n))
    @cuda threads=256 blocks=1 finalize_sum_kernel!(output,Int32(grids))
end
function main(args)
    repeat=length(args)>0 ? parse(Int,args[1]) : 10
    n=length(args)>1 ? parse(Int,args[2]) : 4096
    grids=cld(n,256);input=CUDA.fill(-1f0,n);output=CUDA.zeros(Float32,grids);counter=CUDA.zeros(UInt32,1)
    launch(input,output,counter,n,grids);CUDA.synchronize()
    expected=fill(-256f0,grids);expected[end]=-Float32(n-(grids-1)*256);expected[1]=-Float32(n);push!(expected,0f0)
    for trial in 1:5
        fill!(output,0f0);fill!(counter,0);CUDA.synchronize();start=time_ns()
        for _ in 1:repeat
            launch(input,output,counter,n,grids)
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
        result=vcat(Array(output),Float32.(Array(counter)));error=maximum(abs.(result.-expected));correct=result==expected
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
         ",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"synchronized_complete_reduction_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))

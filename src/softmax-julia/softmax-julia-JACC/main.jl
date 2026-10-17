using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function softmax_jacc!(index,dest,src,numSlice::Int32,sliceSize::Int32)
    i=Int32(index)
    if i<=numSlice
    base = (i - Int32(1)) * sliceSize
    @inbounds begin
        max_ = src[base + Int32(1)]
        j = Int32(1)
        while j <= sliceSize
            v = src[base + j]
            if v > max_
                max_ = v
            end
            j += Int32(1)
        end
        sum = 0.0f0
        j = Int32(1)
        while j <= sliceSize
            sum += exp(src[base + j] - max_)
            j += Int32(1)
        end
        j = Int32(1)
        while j <= sliceSize
            dest[base + j] = exp(src[base + j] - max_) / sum
            j += Int32(1)
        end
    end
    end
end

function factory(output,input,rows,cols)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,Int(rows),softmax_jacc!,output,input,rows,cols)
end
softmax_study(factory)

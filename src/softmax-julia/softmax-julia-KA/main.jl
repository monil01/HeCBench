using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function softmax_ka!(dest, src, numSlice::Int32, sliceSize::Int32)
    i=Int32(@index(Global,Linear))
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
    kernel=softmax_ka!(get_backend(input),256)
    return ()->kernel(output,input,rows,cols;ndrange=Int(rows))
end
softmax_study(factory)

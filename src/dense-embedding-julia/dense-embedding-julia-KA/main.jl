using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function dense_k1_ka!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = Int32(@index(Group,Linear)) - Int32(1)
    tid = Int32(@index(Local,Linear)) - Int32(1)
    grain_size = Int32(256)
    range = offset[batch_idx + Int32(2)] - offset[batch_idx + Int32(1)]
    idx = tid
    while idx < embedding_dim
        dense_elem = @inbounds dense[batch_idx * embedding_dim + idx + Int32(1)]
        nested_idx = idx
        while nested_idx < range
            out_idx = offset[batch_idx + Int32(1)] + nested_idx + Int32(1)
            @inbounds output[out_idx] = input[out_idx] + dense_elem
            nested_idx += embedding_dim
        end
        idx += grain_size
    end
end

@kernel function dense_k2_ka!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = Int32(@index(Group,Linear)) - Int32(1)
    start = offset[batch_idx + Int32(1)]
    range = offset[batch_idx + Int32(2)] - start
    idx = Int32(@index(Local,Linear)) - Int32(1)
    while idx < embedding_dim
        dense_elem = @inbounds dense[batch_idx * embedding_dim + idx + Int32(1)]
        nested_idx = idx
        while nested_idx < range
            out_idx = start + nested_idx + Int32(1)
            @inbounds output[out_idx] = input[out_idx] + dense_elem
            nested_idx += embedding_dim
        end
        idx += Int32(256)
    end
end

@kernel function dense_k3_ka!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = Int32(@index(Group,Linear)) - Int32(1)
    start = offset[batch_idx + Int32(1)]
    range = offset[batch_idx + Int32(2)] - start
    s = Int32(0)
    while s < range
        idx = s + Int32(@index(Local,Linear)) - Int32(1)
        if idx < range
            input_elem = @inbounds input[start + idx + Int32(1)]
            dense_elem = @inbounds dense[batch_idx * embedding_dim + (idx % embedding_dim) + Int32(1)]
            @inbounds output[start + idx + Int32(1)] = input_elem + dense_elem
        end
        s += Int32(256)
    end
end

function factory(input,dense,output,offsets,mode)
    constructor=mode=="k1" ? dense_k1_ka! : mode=="k2" ? dense_k2_ka! : dense_k3_ka!
    kernel=constructor(get_backend(input),256)
    return ()->kernel(input,dense,output,Int32(768),offsets;ndrange=16*256)
end
dense_study(factory)

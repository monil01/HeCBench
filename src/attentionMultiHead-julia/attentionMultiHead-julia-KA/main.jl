using CUDA, KernelAbstractions
# Named source-default geometry: 256 features/head and9 decoded positions.
# Softmax reduction follows preserved CUDA.jl serial9 ordering; native uses CUB block reduction.
@kernel function mha_ka!(q,k,v,beam_size,n_steps,qk_col,v_col,nhead,scale,threshold,dst)
    dim_per_head = qk_col ÷ nhead
    block0 = (@index(Group,Linear)) - Int32(1)
    tid0 = (@index(Local,Linear)) - Int32(1)
    candidate_id = block0 ÷ nhead
    head_id = block0 % nhead

    buffer = @localmem Float32 (265,)
    sq = buffer
    logits_offset = Int(dim_per_head)

    if tid0 < dim_per_head
        pos = candidate_id * qk_col + head_id * dim_per_head + tid0
        @inbounds sq[Int(tid0) + 1] = q[Int(pos) + 1]
    end
    @synchronize

    if tid0 < n_steps
        summ = Float32(0)
        k_base = candidate_id * qk_col * n_steps + head_id * dim_per_head + tid0 * qk_col
        for i0 in Int32(0):(dim_per_head - Int32(1))
            @inbounds summ += sq[Int(i0) + 1] * k[Int(k_base + i0) + 1]
        end
        @inbounds buffer[logits_offset + Int(tid0) + 1] = summ * scale
    end
    @synchronize

    if tid0 == 0
        max_val = Float32(-1.0f-20)
        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds max_val = max(max_val, buffer[logits_offset + Int(t0) + 1])
        end

        denom = Float32(0)
        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds local_i = buffer[logits_offset + Int(t0) + 1] - max_val
            if local_i < -Float32(threshold)
                local_i = -Float32(threshold)
            end
            local_o = exp(local_i)
            @inbounds buffer[logits_offset + Int(t0) + 1] = local_o
            denom += local_o
        end

        for t0 in Int32(0):(n_steps - Int32(1))
            @inbounds buffer[logits_offset + Int(t0) + 1] /= denom
        end
    end
    @synchronize

    if tid0 < dim_per_head
        summ = Float32(0)
        tid = candidate_id * v_col * n_steps + head_id * dim_per_head + tid0
        for i0 in Int32(0):(n_steps - Int32(1))
            @inbounds summ += buffer[logits_offset + Int(i0) + 1] * v[Int(tid + i0 * v_col) + 1]
        end
        out = candidate_id * v_col + head_id * dim_per_head + tid0
        @inbounds dst[Int(out) + 1] = summ
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mha_study((q,k,v,beam,steps,qcol,vcol,heads,scale,threshold,out)->()->mha_ka!(CUDABackend(),256)(q,k,v,beam,steps,qcol,vcol,heads,scale,threshold,out;ndrange=16384))

using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function log_probs_jacc!(gx,gy,log_probs, logits, ids, lengths, max_input_length::Int32,
                           batch_size::Int32, vocab_size::Int32, vocab_size_padded::Int32)
    tid0 = Int32(mod(gx-1,256))
    step0 = Int32(div(gx-1,256))
    bidx0 = Int32(gy-1)
    if tid0 == Int32(0) && bidx0 < batch_size && step0 < lengths[bidx0 + Int32(1)] - Int32(1)
        step_offset = step0 * vocab_size_padded
        batch_offset = bidx0 * max_input_length * vocab_size_padded
        max_val = -typemax(Float32)
        for i0 in Int32(0):(vocab_size - Int32(1))
            @inbounds max_val = max(max_val, logits[batch_offset + step_offset + i0 + Int32(1)])
        end
        sum_exp = 0.0f0
        for i0 in Int32(0):(vocab_size - Int32(1))
            @inbounds sum_exp += exp(logits[batch_offset + step_offset + i0 + Int32(1)] - max_val)
        end
        idx = step0 + bidx0 * (max_input_length - Int32(1)) + Int32(1)
        token_idx = step0 + Int32(1) + bidx0 * max_input_length + Int32(1)
        @inbounds token = ids[token_idx]
        @inbounds log_probs[idx] = logits[batch_offset + step_offset + token + Int32(1)] -
                                   max_val - log(sum_exp + 1.0f-9)
    end
    return
end

function accumulate_jacc!(gx,cum_log_probs, log_probs, lengths, max_input_length::Int32, batch_size::Int32)
    bidx0 = Int32(div(gx-1,256))
    if mod(gx-1,256) == 0 && bidx0 < batch_size
        length = lengths[bidx0 + Int32(1)]
        acc = 0.0f0
        base = bidx0 * (max_input_length - Int32(1))
        for step0 in Int32(0):(length - Int32(2))
            @inbounds acc += log_probs[base + step0 + Int32(1)]
        end
        @inbounds cum_log_probs[bidx0 + Int32(1)] = acc
    end
    return
end

function factory(out,total,input,ids,lengths)
    spec2=JACC.launch_spec(;threads=(256,1),shmem_size=0,sync=false)
    spec1=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->begin
        JACC.parallel_for(spec2,(31*256,64),log_probs_jacc!,out,input,ids,lengths,Int32(32),Int32(64),Int32(256),Int32(256))
        JACC.parallel_for(spec1,64*256,accumulate_jacc!,total,out,lengths,Int32(32),Int32(64))
    end
end
logprob_study(factory)

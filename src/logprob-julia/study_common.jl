using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function log_probs_cpu(logits, ids, lengths, max_input_length, batch_size, vocab_size, vocab_size_padded)
    log_probs = Vector{Float32}(undef, batch_size * (max_input_length - 1))
    cum = Vector{Float32}(undef, batch_size)
    for b in 0:batch_size-1
        acc = 0.0f0
        for step in 0:lengths[b+1]-2
            offset = b * max_input_length * vocab_size_padded + step * vocab_size_padded
            max_val = maximum(@view logits[offset+1:offset+vocab_size])
            sum_exp = 0.0f0
            for i in 1:vocab_size
                sum_exp += exp(logits[offset+i] - max_val)
            end
            token_idx = step + 1 + b * max_input_length + 1
            idx = step + b * (max_input_length - 1) + 1
            log_probs[idx] = logits[offset + ids[token_idx] + 1] - max_val - log(sum_exp + 1.0f-9)
            acc += log_probs[idx]
        end
        cum[b+1] = acc
    end
    return log_probs, cum
end

function logprob_study(factory)
    maxlen=32;batch=64;vocab=256;repeats=10
    logits=Float32[Float32(mod(17*i,97)-48)/8f0 for i in 0:batch*maxlen*vocab-1]
    ids=Int32[mod(13*i,256) for i in 0:batch*maxlen-1];lengths=fill(Int32(maxlen),batch)
    lp,cum=log_probs_cpu(logits,ids,lengths,maxlen,batch,vocab,vocab)
    input=CuArray(logits);di=CuArray(ids);dl=CuArray(lengths)
    output=CUDA.zeros(Float32,batch*(maxlen-1));total=CUDA.zeros(Float32,batch)
    launch! = factory(output,total,input,di,dl)
    OwnedStudy.run_trials!(launch!,()->(fill!(output,0f0);fill!(total,0f0)),()->vcat(Array(output),Array(total)),vcat(lp,cum),repeats;atol=1e-3,rtol=0.)
end

using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function study_cpu_mha(q, k, v, beam_size, n_steps, qk_col, v_col, nhead, scale, threshold)
    dim_per_head = qk_col ÷ nhead
    dst = Vector{Float32}(undef, beam_size * v_col)
    sq = Vector{Float32}(undef, dim_per_head)
    logits = Vector{Float32}(undef, n_steps)
    dp = Vector{Float32}(undef, n_steps)

    for candidate_id in 0:(beam_size - 1), head_id in 0:(nhead - 1)
        for t in 0:(dim_per_head - 1)
            pos = candidate_id * qk_col + head_id * dim_per_head + t
            sq[t + 1] = q[pos + 1]
        end

        for t in 0:(n_steps - 1)
            summ = Float32(0)
            k_base = candidate_id * qk_col * n_steps + head_id * dim_per_head + t * qk_col
            for i in 0:(dim_per_head - 1)
                summ += sq[i + 1] * k[k_base + i + 1]
            end
            dp[t + 1] = summ * scale
        end

        max_val = Float32(-1.0f-20)
        for t in 1:n_steps
            max_val = max(max_val, dp[t])
        end

        denom = Float32(0)
        for t in 1:n_steps
            local_i = dp[t] - max_val
            if local_i < -Float32(threshold)
                local_i = -Float32(threshold)
            end
            logits[t] = exp(local_i)
            denom += logits[t]
        end

        for t in 1:n_steps
            logits[t] /= denom
        end

        for t in 0:(dim_per_head - 1)
            summ = Float32(0)
            tid = candidate_id * v_col * n_steps + head_id * dim_per_head + t
            for i in 0:(n_steps - 1)
                summ += logits[i + 1] * v[tid + i * v_col + 1]
            end
            dst[candidate_id * v_col + head_id * dim_per_head + t + 1] = summ
        end
    end
    return dst
end


function mha_study(factory)
    beams=4;heads=16;cols=4096;steps=9;scale=0.0625f0;threshold=Int32(64);repeats=10
    q=Float32[Float32(i%17-8)/16f0 for i in 0:beams*cols-1]
    k=Float32[Float32((7*i)%31-15)/32f0 for i in 0:beams*cols*steps-1]
    v=Float32[Float32((11*i)%53-26)/32f0 for i in 0:beams*cols*steps-1]
    expected=study_cpu_mha(q,k,v,beams,steps,cols,cols,heads,scale,threshold)
    output=CUDA.zeros(Float32,beams*cols)
    launch! = factory(CuArray(q),CuArray(k),CuArray(v),Int32(beams),Int32(steps),Int32(cols),Int32(cols),Int32(heads),scale,threshold,output)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end

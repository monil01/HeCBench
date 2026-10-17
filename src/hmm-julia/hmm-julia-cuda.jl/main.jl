using CUDA
function viterbi_kernel!(max_old, mt_state, mt_emit, obs, max_new, path,
                         n_state::Int32, t::Int32)
    istate0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if istate0 >= n_state
        return
    end
    max_prob = 0f0
    max_state = Int32(-1)
    pre = Int32(0)
    while pre < n_state
        p = max_old[pre + Int32(1)] + mt_state[istate0 * n_state + pre + Int32(1)]
        if p > max_prob
            max_prob = p
            max_state = pre
        end
        pre += Int32(1)
    end
    max_new[istate0 + Int32(1)] = max_prob + mt_emit[obs[t + Int32(1)] * n_state + istate0 + Int32(1)]
    path[(t - Int32(1)) * n_state + istate0 + Int32(1)] = max_state
    return
end

function study_prepare(old,state,emit,obs,new,path,n)
    t->(@cuda threads=256 blocks=cld(n,256) viterbi_kernel!(old,state,emit,obs,new,path,Int32(n),Int32(t)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end

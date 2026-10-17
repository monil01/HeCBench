using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function adam_jacc!(index,p, m, v, g, b1::Float32, b2::Float32, eps::Float32,
                     grad_scale::Float32, step_size::Float32,
                     time_step::Int32, vector_size::Int32, decay::Float32)
    i = Int32(index)
    totThreads = Int32(65536)
    j = i
    @inbounds while j <= vector_size
        gj = g[j]
        mj = m[j]
        vj = v[j]
        pj = p[j]
        t = Int32(1)
        while t <= time_step
            scaled_grad = gj / grad_scale
            mj = b1 * mj + (1.0f0 - b1) * scaled_grad
            vj = b2 * vj + (1.0f0 - b2) * scaled_grad * scaled_grad
            m_corrected = mj / (1.0f0 - b1^Float32(t))
            v_corrected = vj / (1.0f0 - b2^Float32(t))
            denom = sqrt(v_corrected + eps)  # ADAM_MODE_0
            update = (m_corrected / denom) + (decay * pj)
            pj -= step_size * update
            t += Int32(1)
        end
        m[j] = mj
        v[j] = vj
        p[j] = pj
        j += totThreads
    end
    return
end

function factory(p,m,v,g,n,steps)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,Int(n),adam_jacc!,p,m,v,g,0.9f0,0.999f0,1f-8,256f0,1f-3,steps,n,0.5f0)
end
adam_study(factory)

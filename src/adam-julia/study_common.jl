using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function reference_cpu!(p, m, v, g, b1, b2, eps, grad_scale, step_size,
                        time_step, vector_size, decay)
    for j in 1:vector_size
        for t in 1:time_step
            scaled_grad = g[j] / grad_scale
            m[j] = b1 * m[j] + (1.0f0 - b1) * scaled_grad
            v[j] = b2 * v[j] + (1.0f0 - b2) * scaled_grad * scaled_grad
            m_corrected = m[j] / (1.0f0 - b1^Float32(t))
            v_corrected = v[j] / (1.0f0 - b2^Float32(t))
            denom = sqrt(v_corrected + eps)
            update = (m_corrected / denom) + (decay * p[j])
            p[j] -= step_size * update
        end
    end
end

function adam_study(factory)
    n=65536;steps=4;repeats=10
    p=Float32[Float32(mod(i,29)-14)/16f0 for i in 0:n-1]
    m=Float32[Float32(mod(i,31)+1)/64f0 for i in 0:n-1]
    v=Float32[Float32(mod(i,17)+1)/32f0 for i in 0:n-1]
    g=Float32[Float32(mod(i,23)-11)/8f0 for i in 0:n-1]
    rp=copy(p);rm=copy(m);rv=copy(v)
    for _ in 1:repeats;reference_cpu!(rp,rm,rv,g,0.9f0,0.999f0,1f-8,256f0,1f-3,steps,n,0.5f0);end
    dp=CuArray(p);dm=CuArray(m);dv=CuArray(v);dg=CuArray(g);launch! = factory(dp,dm,dv,dg,Int32(n),Int32(steps))
    OwnedStudy.run_trials!(launch!,()->(copyto!(dp,p);copyto!(dm,m);copyto!(dv,v)),()->vcat(Array(dp),Array(dm),Array(dv)),vcat(rp,rm,rv),repeats;atol=1e-3,rtol=0.)
end

using CUDA
using Printf

# Julia port of adam-cuda benchmark

function adam_kernel!(p, m, v, g, b1::Float32, b2::Float32, eps::Float32,
                     grad_scale::Float32, step_size::Float32,
                     time_step::Int32, vector_size::Int32, decay::Float32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    totThreads = Int32(gridDim().x) * Int32(blockDim().x)
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

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <vector size> <number of time steps> <repeat>")
        return 1
    end
    vector_size = parse(Int, ARGS[1])
    time_step   = parse(Int, ARGS[2])
    repeat_n    = parse(Int, ARGS[3])

    # Deterministic init using LCG (reproducible)
    state = Ref(UInt64(19937))
    function next_f32()
        state[] = state[] * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32((state[] >> 11) & UInt64(0xFFFFFF)) / Float32(1 << 24)
    end

    m = Vector{Float32}(undef, vector_size)
    v = Vector{Float32}(undef, vector_size)
    g = Vector{Float32}(undef, vector_size)
    p = Vector{Float32}(undef, vector_size)
    r = Vector{Float32}(undef, vector_size)

    for i in 1:vector_size
        m[i] = next_f32()
        v[i] = next_f32()
        g[i] = next_f32()
        pv = next_f32()
        r[i] = pv
        p[i] = pv
    end
    m_ref = copy(m); v_ref = copy(v)

    d_m = CuArray(m); d_v = CuArray(v); d_g = CuArray(g); d_p = CuArray(p)

    step_size = 1f-3
    decay = 0.5f0
    beta1 = 0.9f0
    beta2 = 0.999f0
    eps = 1f-8
    grad_scale = 256.0f0

    threadsPerBlock = 256
    blocks = cld(vector_size, threadsPerBlock)

    # Warmup + time
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threadsPerBlock blocks=blocks adam_kernel!(d_p, d_m, d_v, d_g,
            beta1, beta2, eps, grad_scale, step_size,
            Int32(time_step), Int32(vector_size), decay)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1e-6 / repeat_n
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms)

    p_gpu = Array(d_p)

    for _ in 1:repeat_n
        reference_cpu!(r, m_ref, v_ref, g, beta1, beta2, eps, grad_scale,
                       step_size, time_step, vector_size, decay)
    end

    ok = true
    cr = 0.0
    cp = 0.0
    for i in 1:vector_size
        if abs(r[i] - p_gpu[i]) > 1f-3
            ok = false
            break
        end
        cr += r[i]; cp += p_gpu[i]
    end

    println(ok ? "PASS" : "FAIL")
    @printf("Checksum: %f %f\n", cr / vector_size, cp / vector_size)
    return 0
end

main()

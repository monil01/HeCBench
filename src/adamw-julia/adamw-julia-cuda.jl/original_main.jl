using CUDA
using Printf

# adamw Julia port. Element-wise AdamW: decoupled weight decay from Adam.
# Each thread processes one parameter; inner loop runs `time_step` steps.

function adamw_kernel!(p, m, v, g, b1::Float32, b2::Float32, eps::Float32,
                      grad_scale::Float32, step_size::Float32, decay::Float32,
                      vector_size::Int32, time_step::Int32)
    j = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    if j > vector_size
        return
    end
    @inbounds begin
        sg = g[j] / grad_scale
        mj = m[j]
        vj = v[j]
        pj = p[j]
        for t in 1:time_step
            mj = b1 * mj + (1f0 - b1) * sg
            vj = b2 * vj + (1f0 - b2) * sg * sg
            m_hat = mj / (1f0 - b1^Float32(t))
            v_hat = vj / (1f0 - b2^Float32(t))
            # AdamW: decoupled weight decay
            pj = pj - step_size * (m_hat / (sqrt(v_hat) + eps) + decay * pj)
        end
        p[j] = pj; m[j] = mj; v[j] = vj
    end
    return
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
    time_step = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    repeat = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

    b1 = 0.9f0; b2 = 0.999f0; eps = 1f-8; grad_scale = 1f0
    lr = 1f-3; decay = 1f-2

    p = ones(Float32, n) .* 0.5f0
    m = zeros(Float32, n)
    v = zeros(Float32, n)
    g = Float32.((1:n) .% 7) .* 0.01f0

    # CPU reference before device runs
    pref = copy(p); mref = copy(m); vref = copy(v)
    for _ in 1:repeat
        for j in 1:n
            sg = g[j] / grad_scale
            for t in 1:time_step
                mref[j] = b1 * mref[j] + (1f0 - b1) * sg
                vref[j] = b2 * vref[j] + (1f0 - b2) * sg * sg
                m_hat = mref[j] / (1f0 - b1^Float32(t))
                v_hat = vref[j] / (1f0 - b2^Float32(t))
                pref[j] = pref[j] - lr * (m_hat / (sqrt(v_hat) + eps) + decay * pref[j])
            end
        end
    end

    dp = CuArray(p); dm = CuArray(m); dv = CuArray(v); dg = CuArray(g)
    block = 256
    grid = cld(n, block)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid adamw_kernel!(dp, dm, dv, dg,
            b1, b2, eps, grad_scale, lr, decay, Int32(n), Int32(time_step))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) / 1e6 / repeat
    @printf "Average kernel execution time: %.3f (ms)\n" elapsed_ms

    pgpu = Array(dp)
    maxabs = maximum(abs.(pgpu .- pref))
    @printf "max |err| = %g\n" maxabs
    println(maxabs < 1f-3 ? "PASS" : "FAIL")
end

main()

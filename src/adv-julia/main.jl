using CUDA
using Printf

# adv Julia port: 1D advection surrogate. u[i] += g[i]*(u[i-1] - u[i+1])
# with periodic boundaries, run iters times.

function adv_kernel!(unext, u, g, n::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    if i > n
        return
    end
    im = (i - Int32(2) + n) % n + Int32(1)
    ip = i % n + Int32(1)
    @inbounds unext[i] = u[i] + g[i] * (u[im] - u[ip])
    return
end

function main()
    n = 65536
    iters = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    u = zeros(Float32, n)
    g = zeros(Float32, n)
    s = UInt64(20260721)
    for i in 1:n
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        u[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        g[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff) * 0.01f0
    end

    du = CuArray(u)
    dg = CuArray(g)
    dn = CuArray(zeros(Float32, n))

    block = 256
    grid = cld(n, block)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iters
        @cuda threads=block blocks=grid adv_kernel!(dn, du, dg, Int32(n))
        du, dn = dn, du
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) / 1e3 / iters
    @printf "elapsed time= %.3f us/iter\n" elapsed_us

    # Host reference
    uref = copy(u)
    tmp = zeros(Float32, n)
    for _ in 1:iters
        for i in 1:n
            im = (i - 2 + n) % n + 1
            ip = i % n + 1
            tmp[i] = uref[i] + g[i] * (uref[im] - uref[ip])
        end
        uref, tmp = tmp, uref
    end

    gpu = Array(du)
    maxabs = maximum(abs.(gpu .- uref))
    @printf "Max error: %g\n" maxabs
    println(maxabs <= 1f-3 ? "PASS" : "FAIL")
end

main()

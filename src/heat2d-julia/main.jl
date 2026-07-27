using CUDA
using Printf

# Julia port of heat2d-cuda benchmark. Periodic 2D Laplacian relaxation
# on an Lx*Ly grid. GPU kernel is straight port; CPU reference verifies.

const NTX = 16
const NTY = 16

function lapl_kernel!(out, in_, delta::Float32, norm::Float32, Lx::Int32, Ly::Int32)
    tid = Int32(threadIdx().x)  # 1..NTX*NTY
    i = (Int32(blockIdx().x - 1)) * Int32(blockDim().x) + tid - Int32(1)  # 0-based
    N = Lx * Ly
    if i < N
        x = i % Lx
        y = i ÷ Lx
        v00 = y * Lx + x
        v0p = y * Lx + ((x + Int32(1)) % Lx)
        v0m = y * Lx + ((Lx + x - Int32(1)) % Lx)
        vp0 = ((y + Int32(1)) % Ly) * Lx + x
        vm0 = ((Ly + y - Int32(1)) % Ly) * Lx + x
        @inbounds out[v00 + Int32(1)] = norm * in_[v00 + Int32(1)] +
            delta * (in_[v0p + Int32(1)] + in_[v0m + Int32(1)] +
                     in_[vp0 + Int32(1)] + in_[vm0 + Int32(1)])
    end
    return
end

# Simple LCG to reproduce srand(123) style seeds -- doesn't need to match glibc
# exactly, only needs to be deterministic. CPU and GPU use SAME buffer so verify
# still works.
mutable struct LCG
    s::UInt64
end
LCG(seed) = LCG(UInt64(seed))
function next_int!(l::LCG, mod)
    l.s = l.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Int((l.s >> 33) % UInt64(mod))
end

function reference!(out, in_, delta, norm, Lx, Ly)
    for y in 0:(Ly-1), x in 0:(Lx-1)
        v00 = y*Lx + x
        v0p = y*Lx + ((x + 1) % Lx)
        v0m = y*Lx + ((Lx + x - 1) % Lx)
        vp0 = ((y + 1) % Ly) * Lx + x
        vm0 = ((Ly + y - 1) % Ly) * Lx + x
        out[v00 + 1] = norm * in_[v00 + 1] +
            delta * (in_[v0p + 1] + in_[v0m + 1] + in_[vp0 + 1] + in_[vm0 + 1])
    end
end

function main()
    if length(ARGS) < 3
        println(" Usage: main.jl LX LY NITER")
        return 1
    end
    Lx = parse(Int, ARGS[1])
    Ly = parse(Int, ARGS[2])
    niter = parse(Int, ARGS[3])
    if Lx % NTX != 0 || Ly % NTY != 0
        @printf("Array length LX and LY must be a multiple of block size %d and %d, respectively\n", NTX, NTY)
        return 1
    end

    sigma = 0.01f0
    xdelta = sigma / (1.0f0 + 4.0f0 * sigma)
    xnorm  = 1.0f0 / (1.0f0 + 4.0f0 * sigma)

    @printf(" Ly,Lx = %d,%d\n", Ly, Lx)
    @printf(" niter = %d\n", niter)

    lcg = LCG(123)
    buffer = zeros(Float32, Lx * Ly)
    for i in 0:16:(Lx-1)
        x = next_int!(lcg, Lx)
        for j in 0:(Ly-1)
            buffer[x + j*Lx + 1] = 1f0
        end
    end
    for i in 0:16:(Ly-1)
        y = next_int!(lcg, Ly)
        for j in 0:(Lx-1)
            buffer[j + y*Lx + 1] = 1f0
        end
    end

    h_in = copy(buffer)
    h_out = similar(buffer)
    for _ in 1:niter
        reference!(h_out, h_in, xdelta, xnorm, Lx, Ly)
        h_in, h_out = h_out, h_in
    end

    d_in = CuArray(buffer)
    d_out = CUDA.zeros(Float32, Lx * Ly)

    threads = NTX * NTY
    blocks = cld(Lx * Ly, threads)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:niter
        @cuda threads=threads blocks=blocks lapl_kernel!(d_out, d_in, xdelta, xnorm,
                                                          Int32(Lx), Int32(Ly))
        d_in, d_out = d_out, d_in
    end
    CUDA.synchronize()
    tus = (time_ns() - t0) * 1e-3 / niter
    bw = Lx*Ly*sizeof(Float32)*2.0/(tus*1e3)
    gflops = (Lx*Ly*6.0)/(tus*1e3)
    @printf("Device: iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
            niter, Lx, Ly, tus, bw, gflops)

    d_res = Array(d_in)
    ok = true
    for i in 1:Lx*Ly
        if abs(h_in[i] - d_res[i]) > 1e-2
            @printf("Mismatch at %d cpu=%f gpu=%f\n", i-1, h_in[i], d_res[i])
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()

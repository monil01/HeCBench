using CUDA
using Printf

# hotspot Julia port. 5-point stencil update: temp evolves under power
# input + Laplacian. Uses synthetic deterministic data (upstream needs
# temp_512/power_512 DVC files) and CUDA benchmark's declared tolerance.

const L = 512
const ITERS = 200

function hotspot_kernel!(out, cur, power, step_div_Cap::Float32,
                       Rx_1::Float32, Ry_1::Float32, Rz_1::Float32)
    x = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    y = Int32((blockIdx().y - 1)) * Int32(blockDim().y) + Int32(threadIdx().y)
    if x > L || y > L
        return
    end
    N = y - Int32(1); if N < Int32(1); N = Int32(1); end
    S = y + Int32(1); if S > Int32(L); S = Int32(L); end
    W = x - Int32(1); if W < Int32(1); W = Int32(1); end
    E = x + Int32(1); if E > Int32(L); E = Int32(L); end
    @inbounds begin
        idx = (y - 1) * L + x
        t = cur[idx]
        out[idx] = t + step_div_Cap * (power[idx]
            + (cur[(S - 1) * L + x] + cur[(N - 1) * L + x] - 2f0*t) * Ry_1
            + (cur[(y - 1) * L + E] + cur[(y - 1) * L + W] - 2f0*t) * Rx_1
            + (80f0 - t) * Rz_1)
    end
    return
end

function main()
    L2 = L * L
    # Synthesise deterministic input
    tvec = Vector{Float32}(undef, L2)
    pvec = Vector{Float32}(undef, L2)
    s = UInt64(20250723)
    for i in 1:L2
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        tvec[i] = 300f0 + Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff) * 50f0
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        pvec[i] = Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff) * 0.5f0
    end

    # Physical constants (from hotspot.h)
    chip_height = 0.016f0; chip_width = 0.016f0
    grid_h = chip_height / L; grid_w = chip_width / L
    t_chip = 0.0005f0; K_SI = 100f0; SPEC_HEAT_SI = 1.75f6; FACTOR_CHIP = 0.5f0
    MAX_PD = 3.0f6; PRECISION = 0.001f0
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_w * grid_h
    Rx = grid_w / (2f0 * K_SI * t_chip * grid_h)
    Ry = grid_h / (2f0 * K_SI * t_chip * grid_w)
    Rz = t_chip / (K_SI * grid_h * grid_w)
    step = PRECISION / (MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI))
    step_div_Cap = step / Cap
    Rx_1 = 1f0/Rx; Ry_1 = 1f0/Ry; Rz_1 = 1f0/Rz

    d_p = CuArray(pvec)
    d_a = CuArray(tvec)
    d_b = CuArray(zeros(Float32, L2))
    BLK = 16
    grid = (cld(L, BLK), cld(L, BLK))

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:ITERS
        @cuda threads=(BLK, BLK) blocks=grid hotspot_kernel!(d_b, d_a, d_p,
                                                            step_div_Cap, Rx_1, Ry_1, Rz_1)
        d_a, d_b = d_b, d_a
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) / 1e9
    @printf "Total kernel execution time %f (s)\n" elapsed

    # CPU reference
    cur = copy(tvec); nxt = zeros(Float32, L2)
    for _ in 1:ITERS
        for y in 1:L, x in 1:L
            N = max(y - 1, 1); S = min(y + 1, L)
            W = max(x - 1, 1); E = min(x + 1, L)
            idx = (y - 1) * L + x
            t = cur[idx]
            nxt[idx] = t + step_div_Cap * (pvec[idx]
                + (cur[(S - 1) * L + x] + cur[(N - 1) * L + x] - 2f0*t) * Ry_1
                + (cur[(y - 1) * L + E] + cur[(y - 1) * L + W] - 2f0*t) * Rx_1
                + (80f0 - t) * Rz_1)
        end
        cur, nxt = nxt, cur
    end

    gpu = Array(d_a)
    maxabs = maximum(abs.(gpu .- cur))
    @printf "max |err| = %g\n" maxabs
    println(maxabs < 1f-3 ? "PASS" : "FAIL")
end

main()

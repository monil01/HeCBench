using CUDA
using Printf
using Random

const LN_2 = Float32(0.6931471805599453)
const TWO_TO_MINUS_31 = Float32(4.6566128730773926e-10)
const TWO_TO_MINUS_32 = Float32(2.3283064365386963e-10)
const TEA_K0 = UInt32(0xa341316c)
const TEA_K1 = UInt32(0xc8013ea4)
const TEA_K2 = UInt32(0xad90777d)
const TEA_K3 = UInt32(0x7e95761e)
const TEA_DT = UInt32(0x9e3779b9)

@inline function minimum_image(dr::Float32, p::Float32)
    half = p * 0.5f0
    return dr + (dr > -half ? (dr < half ? 0.0f0 : -p) : p)
end

@inline function tea_rounds(v0::UInt32, v1::UInt32)
    sum = UInt32(0)
    @inbounds for _ in 1:4
        sum += TEA_DT
        v0 += ((v1 << 4) + TEA_K0) ⊻ (v1 + sum) ⊻ ((v1 >> 5) + TEA_K1)
        v1 += ((v0 << 4) + TEA_K2) ⊻ (v0 + sum) ⊻ ((v0 >> 5) + TEA_K3)
    end
    return v0, v1
end

@inline function gaussian_tea_fast(pred::Bool, u::Int32, v::Int32)
    v0 = pred ? UInt32(u) : UInt32(v)
    v1 = pred ? UInt32(v) : UInt32(u)
    v0, v1 = tea_rounds(v0, v1)
    f = sinpi(Float32(reinterpret(Int32, v0)) * TWO_TO_MINUS_31)
    x = max(Float32(v1) * TWO_TO_MINUS_32, eps(Float32))
    r = sqrt(-2.0f0 * LN_2 * log2(x))
    return min(4.0f0, max(-4.0f0, r * f))
end

@inline fpow(x::Float32, y::Float32) = exp(y * log(x))

function bond_kernel!(
    force_x, force_y, force_z,
    coord_x, coord_y, coord_z,
    veloc_x, veloc_y, veloc_z, veloc_w,
    nbond, bond_type, bond_r0,
    temp, r0, mu_targ, qp, gamc, gamt, sigc, sigt,
    period_x::Float32, period_y::Float32, period_z::Float32,
    n_type::Int32, n_local::Int32)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    total = n_local + Int32(1)

    while idx <= n_local
        i = idx + Int32(1)
        n = nbond[i]
        cx = coord_x[i]
        cy = coord_y[i]
        cz = coord_z[i]
        vx = veloc_x[i]
        vy = veloc_y[i]
        vz = veloc_z[i]
        vw = veloc_w[i]
        fxi = 0.0f0
        fyi = 0.0f0
        fzi = 0.0f0

        @inbounds for p in Int32(0):(n - Int32(1))
            flat = idx + p
            j = (flat + Int32(1)) % total + Int32(1)
            typ = bond_type[flat + Int32(1)] + Int32(1)

            delx = minimum_image(cx - coord_x[j], period_x)
            dely = minimum_image(cy - coord_y[j], period_y)
            delz = minimum_image(cz - coord_z[j], period_z)
            dvx = vx - veloc_x[j]
            dvy = vy - veloc_y[j]
            dvz = vz - veloc_z[j]

            l0 = Float32(bond_r0[flat + Int32(1)])
            r0t = r0[typ]
            qpt = qp[typ]
            tempt = temp[typ]
            lmax = l0 * r0t
            rr_inv = 1.0f0 / r0t
            sr = (1.0f0 - rr_inv) * (1.0f0 - rr_inv)
            kph = fpow(l0, qpt) * tempt * (0.25f0 / sr - 0.25f0 + rr_inv)
            mu = 0.433f0 * (tempt * (-0.25f0 / sr + 0.25f0 + 0.5f0 * rr_inv / (sr * (1.0f0 - rr_inv))) /
                 (lmax * rr_inv) + kph * (qpt + 1.0f0) / fpow(l0, qpt + 1.0f0))
            lambda = mu / mu_targ[typ]
            kph /= lambda
            ra = sqrt(delx * delx + dely * dely + delz * delz)
            rr = min(0.99f0, ra / lmax)
            rlogarg = max(0.01f0, fpow(ra, qpt + 1.0f0))
            vv = (delx * dvx + dely * dvy + delz * dvz) / ra

            v1 = Int32(round(vw * 100000.0f0))
            v2 = Int32(round(veloc_w[j] * 100000.0f0))
            ww00 = gaussian_tea_fast(v1 > v2, v1, v2)
            ww01 = gaussian_tea_fast(v1 > v2, v1, v2 + Int32(1))
            ww02 = gaussian_tea_fast(v1 > v2, v1, v2 + Int32(2))
            ww10 = gaussian_tea_fast(v1 > v2, v1 + Int32(1), v2)
            ww11 = gaussian_tea_fast(v1 > v2, v1 + Int32(1), v2 + Int32(1))
            ww12 = gaussian_tea_fast(v1 > v2, v1 + Int32(1), v2 + Int32(2))
            ww20 = gaussian_tea_fast(v1 > v2, v1 + Int32(2), v2)
            ww21 = gaussian_tea_fast(v1 > v2, v1 + Int32(2), v2 + Int32(1))
            ww22 = gaussian_tea_fast(v1 > v2, v1 + Int32(2), v2 + Int32(2))
            w = (ww00 + ww11 + ww22) / 3.0f0
            wrx = (ww00 - w) * delx + 0.5f0 * (ww01 + ww10) * dely + 0.5f0 * (ww02 + ww20) * delz
            wry = 0.5f0 * (ww10 + ww01) * delx + (ww11 - w) * dely + 0.5f0 * (ww12 + ww21) * delz
            wrz = 0.5f0 * (ww20 + ww02) * delx + 0.5f0 * (ww21 + ww12) * dely + (ww22 - w) * delz

            fforce = -tempt * (0.25f0 / (1.0f0 - rr) / (1.0f0 - rr) - 0.25f0 + rr) / lambda / ra +
                     kph / rlogarg + (sigc[typ] * w - gamc[typ] * vv) / ra
            fxi += delx * fforce - gamt[typ] * dvx + sigt[typ] * wrx / ra
            fyi += dely * fforce - gamt[typ] * dvy + sigt[typ] * wry / ra
            fzi += delz * fforce - gamt[typ] * dvz + sigt[typ] * wrz / ra
        end

        force_x[i] += Float64(fxi)
        force_y[i] += Float64(fyi)
        force_z[i] += Float64(fzi)
        idx += stride
    end
    return
end

function main()
    if !(length(ARGS) in (1, 2))
        println("Usage: main.jl <repeat> [n]")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    n = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 1_000_000
    n_type = 32
    repeat > 0 || error("repeat must be positive")
    n > 0 || error("n must be positive")
    CUDA.allowscalar(false)

    rng = MersenneTwister(19937)
    total = n + 1
    extended = n + n + 1
    rand32() = Float32(rand(rng) * 0.8 + 0.1)
    rand64() = Float64(rand(rng) * 0.8 + 0.1)

    bond_r0 = [rand64() + 0.001 for _ in 1:extended]
    bond_type = Int32.(rand(rng, 0:n_type, extended))
    nbond = Int32.(rand(rng, 0:n_type, total))
    coord_x = [rand32() for _ in 1:total]
    coord_y = [rand32() for _ in 1:total]
    coord_z = [rand32() for _ in 1:total]
    veloc_x = [rand32() for _ in 1:total]
    veloc_y = [rand32() for _ in 1:total]
    veloc_z = [rand32() for _ in 1:total]
    veloc_w = sqrt.(veloc_x .* veloc_x .+ veloc_y .* veloc_y .+ veloc_z .* veloc_z)

    temp = [rand32() for _ in 1:(n_type + 1)]
    mu_targ = [rand32() for _ in 1:(n_type + 1)]
    qp = [rand32() for _ in 1:(n_type + 1)]
    gamt = [rand32() for _ in 1:(n_type + 1)]
    gamc = [Float32(((rand(rng, 0:n_type) % 4) + 4) * gamt[i]) for i in 1:(n_type + 1)]
    r0 = [rand32() for _ in 1:(n_type + 1)]
    sigc = sqrt.(2.0f0 .* temp .* (3.0f0 .* gamc .- gamt))
    sigt = 2.0f0 .* sqrt.(gamt .* temp)

    force_x = CUDA.zeros(Float64, total)
    force_y = CUDA.zeros(Float64, total)
    force_z = CUDA.zeros(Float64, total)
    threads = 128
    blocks = cld(n, threads)

    d_args = (
        force_x, force_y, force_z,
        CuArray(coord_x), CuArray(coord_y), CuArray(coord_z),
        CuArray(veloc_x), CuArray(veloc_y), CuArray(veloc_z), CuArray(veloc_w),
        CuArray(nbond), CuArray(bond_type), CuArray(bond_r0),
        CuArray(temp), CuArray(r0), CuArray(mu_targ), CuArray(qp),
        CuArray(gamc), CuArray(gamt), CuArray(sigc), CuArray(sigt),
        0.5f0, 0.5f0, 0.5f0, Int32(n_type), Int32(n)
    )

    synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks bond_kernel!(d_args...)
    end
    synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (us)\n", elapsed * 1.0e-3 / repeat)

    hx = Array(force_x)
    hy = Array(force_y)
    hz = Array(force_z)
    ok = true
    for i in 1:total
        bad = isnan(hx[i]) || isnan(hy[i]) || isnan(hz[i])
        if bad
            @printf("There are NaN numbers at index %d\n", i - 1)
            ok = false
            break
        end
    end
    @printf("checksum: forceX=%lf forceY=%lf forceZ=%lf\n", sum(hx) / total, sum(hy) / total, sum(hz) / total)
    println(ok ? "PASS" : "FAIL")
    ok || exit(1)
end

main()

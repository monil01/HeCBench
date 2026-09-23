using CUDA
using Printf

const BLOCK_SIZE = 64
const GRID_SIZE = 1500
const NN = 80
const NMAT = 40
const L2_SIZE = 3280
const NOPT = 15
const NPATH = 96000

@inline fdiv(a::Float32, b::Float32) = a / b

function path_calc!(L, z, lambda, delta::Float32)
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        sqez = sqrt(delta) * z[Int(n0) + 1]
        v = Float32(0)
        i0 = n0 + Int32(1)
        while i0 < Int32(NN)
            lam = @inbounds lambda[Int(i0 - n0)]
            con1 = delta * lam
            Li = L[Int(i0) + 1]
            v += fdiv(con1 * Li, Float32(1) + delta * Li)
            vrat = exp(con1 * v + lam * (sqez - Float32(0.5) * con1))
            L[Int(i0) + 1] = Li * vrat
            i0 += Int32(1)
        end
    end
    return
end

function path_calc_b1!(L, z, L2, lambda, delta::Float32)
    for i in 1:NN
        L2[i] = L[i]
    end
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        sqez = sqrt(delta) * z[Int(n0) + 1]
        v = Float32(0)
        i0 = n0 + Int32(1)
        while i0 < Int32(NN)
            lam = @inbounds lambda[Int(i0 - n0)]
            con1 = delta * lam
            Li = L[Int(i0) + 1]
            v += fdiv(con1 * Li, Float32(1) + delta * Li)
            vrat = exp(con1 * v + lam * (sqez - Float32(0.5) * con1))
            L[Int(i0) + 1] = Li * vrat
            L2[Int(i0 + (n0 + Int32(1)) * Int32(NN)) + 1] = L[Int(i0) + 1]
            i0 += Int32(1)
        end
    end
    return
end

function path_calc_b2!(L_b, z, L2, lambda, delta::Float32)
    n0 = Int32(NMAT - 1)
    while n0 >= 0
        v1 = Float32(0)
        i0 = Int32(NN - 1)
        while i0 > n0
            lam = @inbounds lambda[Int(i0 - n0)]
            l2_next = L2[Int(i0 + (n0 + Int32(1)) * Int32(NN)) + 1]
            l2_cur = L2[Int(i0 + n0 * Int32(NN)) + 1]
            v1 += lam * l2_next * L_b[Int(i0) + 1]
            faci = fdiv(delta, Float32(1) + delta * l2_cur)
            L_b[Int(i0) + 1] =
                L_b[Int(i0) + 1] * fdiv(l2_next, l2_cur) + v1 * lam * faci * faci
            i0 -= Int32(1)
        end
        n0 -= Int32(1)
    end
    return
end

function portfolio!(L, lambda, maturities, swaprates, delta::Float32)
    B = MVector{NMAT,Float32}(undef)
    S = MVector{NMAT,Float32}(undef)
    b = Float32(1)
    s = Float32(0)
    for n in NMAT:(NN - 1)
        b = b / (Float32(1) + delta * L[n + 1])
        s += delta * b
        B[n - NMAT + 1] = b
        S[n - NMAT + 1] = s
    end

    v = Float32(0)
    for i in 1:NOPT
        m = @inbounds maturities[i]
        swapval = B[Int(m)] + (@inbounds swaprates[i]) * S[Int(m)] - Float32(1)
        if swapval < Float32(0)
            v += Float32(-100) * swapval
        end
    end

    b = Float32(1)
    for n in 1:NMAT
        b = b / (Float32(1) + delta * L[n])
    end
    return b * v
end

function portfolio_b!(L, L_b, lambda, maturities, swaprates, delta::Float32)
    B = MVector{NMAT,Float32}(undef)
    S = MVector{NMAT,Float32}(undef)
    B_b = MVector{NMAT,Float32}(undef)
    S_b = MVector{NMAT,Float32}(undef)

    b = Float32(1)
    s = Float32(0)
    for m0 in 0:(NN - NMAT - 1)
        n = m0 + NMAT
        b = fdiv(b, Float32(1) + delta * L[n + 1])
        s += delta * b
        B[m0 + 1] = b
        S[m0 + 1] = s
    end

    v = Float32(0)
    for m in 1:NMAT
        B_b[m] = Float32(0)
        S_b[m] = Float32(0)
    end

    for n in 1:NOPT
        m = @inbounds maturities[n]
        swapval = B[Int(m)] + (@inbounds swaprates[n]) * S[Int(m)] - Float32(1)
        if swapval < Float32(0)
            v += Float32(-100) * swapval
            S_b[Int(m)] += Float32(-100) * (@inbounds swaprates[n])
            B_b[Int(m)] += Float32(-100)
        end
    end

    m0 = NN - NMAT - 1
    while m0 >= 0
        n = m0 + NMAT
        B_b[m0 + 1] += delta * S_b[m0 + 1]
        L_b[n + 1] = -B_b[m0 + 1] * B[m0 + 1] *
                     fdiv(delta, Float32(1) + delta * L[n + 1])
        if m0 > 0
            S_b[m0] += S_b[m0 + 1]
            B_b[m0] += fdiv(B_b[m0 + 1], Float32(1) + delta * L[n + 1])
        end
        m0 -= 1
    end

    b = Float32(1)
    for n in 1:NMAT
        b = b / (Float32(1) + delta * L[n])
    end
    v = b * v

    for n in 1:NMAT
        L_b[n] = -v * delta / (Float32(1) + delta * L[n])
    end
    for n in (NMAT + 1):NN
        L_b[n] = b * L_b[n]
    end
    return v
end

@inline lidx(base::Int64, i0::Int32) = base + Int64(i0) + Int64(1)
@inline l2idx(base::Int64, i0::Int32, n0::Int32) = base + Int64(i0) + Int64(n0) * Int64(NN) + Int64(1)

function path_calc_global!(L, base::Int64, lambda, delta::Float32)
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        sqez = sqrt(delta) * Float32(0.3)
        v = Float32(0)
        i0 = n0 + Int32(1)
        while i0 < Int32(NN)
            lam = @inbounds lambda[Int(i0 - n0)]
            con1 = delta * lam
            Li = @inbounds L[lidx(base, i0)]
            v += fdiv(con1 * Li, Float32(1) + delta * Li)
            vrat = exp(con1 * v + lam * (sqez - Float32(0.5) * con1))
            @inbounds L[lidx(base, i0)] = Li * vrat
            i0 += Int32(1)
        end
    end
    return
end

function path_calc_b1_global!(L, base::Int64, L2, l2base::Int64, lambda, delta::Float32)
    for i0 in Int32(0):(Int32(NN) - Int32(1))
        @inbounds L2[l2idx(l2base, i0, Int32(0))] = L[lidx(base, i0)]
    end
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        sqez = sqrt(delta) * Float32(0.3)
        v = Float32(0)
        i0 = n0 + Int32(1)
        while i0 < Int32(NN)
            lam = @inbounds lambda[Int(i0 - n0)]
            con1 = delta * lam
            Li = @inbounds L[lidx(base, i0)]
            v += fdiv(con1 * Li, Float32(1) + delta * Li)
            vrat = exp(con1 * v + lam * (sqez - Float32(0.5) * con1))
            Li = Li * vrat
            @inbounds L[lidx(base, i0)] = Li
            @inbounds L2[l2idx(l2base, i0, n0 + Int32(1))] = Li
            i0 += Int32(1)
        end
    end
    return
end

function path_calc_b2_global!(L_b, base::Int64, L2, l2base::Int64, lambda, delta::Float32)
    n0 = Int32(NMAT - 1)
    while n0 >= 0
        v1 = Float32(0)
        i0 = Int32(NN - 1)
        while i0 > n0
            lam = @inbounds lambda[Int(i0 - n0)]
            l2_next = @inbounds L2[l2idx(l2base, i0, n0 + Int32(1))]
            l2_cur = @inbounds L2[l2idx(l2base, i0, n0)]
            Lbi = @inbounds L_b[lidx(base, i0)]
            v1 += lam * l2_next * Lbi
            faci = fdiv(delta, Float32(1) + delta * l2_cur)
            @inbounds L_b[lidx(base, i0)] = Lbi * fdiv(l2_next, l2_cur) + v1 * lam * faci * faci
            i0 -= Int32(1)
        end
        n0 -= Int32(1)
    end
    return
end

@inline widx(base::Int64, i0::Int32) = base + Int64(i0) + Int64(1)

function portfolio_global!(L, base::Int64, lambda, maturities, swaprates, delta::Float32,
                           B, S, wbase::Int64)
    b = Float32(1)
    s = Float32(0)
    for n in NMAT:(NN - 1)
        b = b / (Float32(1) + delta * (@inbounds L[lidx(base, Int32(n))]))
        s += delta * b
        j0 = Int32(n - NMAT)
        @inbounds B[widx(wbase, j0)] = b
        @inbounds S[widx(wbase, j0)] = s
    end
    v = Float32(0)
    for i in 1:NOPT
        m = @inbounds maturities[i]
        mi0 = m - Int32(1)
        swapval = (@inbounds B[widx(wbase, mi0)]) +
                  (@inbounds swaprates[i]) * (@inbounds S[widx(wbase, mi0)]) - Float32(1)
        if swapval < Float32(0)
            v += Float32(-100) * swapval
        end
    end
    b = Float32(1)
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        b = b / (Float32(1) + delta * (@inbounds L[lidx(base, n0)]))
    end
    return b * v
end

function portfolio_b_global!(L, base::Int64, lambda, maturities, swaprates, delta::Float32,
                             B, S, B_b, S_b, wbase::Int64)
    b = Float32(1)
    s = Float32(0)
    for m0 in 0:(NN - NMAT - 1)
        n = m0 + NMAT
        b = fdiv(b, Float32(1) + delta * (@inbounds L[lidx(base, Int32(n))]))
        s += delta * b
        @inbounds B[widx(wbase, Int32(m0))] = b
        @inbounds S[widx(wbase, Int32(m0))] = s
        @inbounds B_b[widx(wbase, Int32(m0))] = Float32(0)
        @inbounds S_b[widx(wbase, Int32(m0))] = Float32(0)
    end
    v = Float32(0)
    for n in 1:NOPT
        m = @inbounds maturities[n]
        swaprate = @inbounds swaprates[n]
        mi0 = m - Int32(1)
        swapval = (@inbounds B[widx(wbase, mi0)]) + swaprate * (@inbounds S[widx(wbase, mi0)]) - Float32(1)
        if swapval < Float32(0)
            v += Float32(-100) * swapval
            @inbounds S_b[widx(wbase, mi0)] = S_b[widx(wbase, mi0)] + Float32(-100) * swaprate
            @inbounds B_b[widx(wbase, mi0)] = B_b[widx(wbase, mi0)] + Float32(-100)
        end
    end
    m0 = NN - NMAT - 1
    while m0 >= 0
        n = m0 + NMAT
        m0i = Int32(m0)
        @inbounds B_b[widx(wbase, m0i)] = B_b[widx(wbase, m0i)] + delta * S_b[widx(wbase, m0i)]
        Ln = @inbounds L[lidx(base, Int32(n))]
        @inbounds L[lidx(base, Int32(n))] =
            -B_b[widx(wbase, m0i)] * B[widx(wbase, m0i)] * fdiv(delta, Float32(1) + delta * Ln)
        if m0 > 0
            prev = Int32(m0 - 1)
            @inbounds S_b[widx(wbase, prev)] = S_b[widx(wbase, prev)] + S_b[widx(wbase, m0i)]
            @inbounds B_b[widx(wbase, prev)] = B_b[widx(wbase, prev)] +
                                               fdiv(B_b[widx(wbase, m0i)], Float32(1) + delta * Ln)
        end
        m0 -= 1
    end
    b = Float32(1)
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        b = b / (Float32(1) + delta * (@inbounds L[lidx(base, n0)]))
    end
    v = b * v
    for n0 in Int32(0):(Int32(NMAT) - Int32(1))
        Ln = @inbounds L[lidx(base, n0)]
        @inbounds L[lidx(base, n0)] = -v * delta / (Float32(1) + delta * Ln)
    end
    for n0 in Int32(NMAT):(Int32(NN) - Int32(1))
        @inbounds L[lidx(base, n0)] = b * L[lidx(base, n0)]
    end
    return v
end

function pathcalc_portfolio_kernel!(d_v, d_Lb, lambda, maturities, swaprates, delta::Float32,
                                    Lwork, L2work, Bwork, Swork, Bbwork, Sbwork)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    threadN = blockDim().x * gridDim().x
    path = tid0
    while path < Int32(NPATH)
        p = Int64(path)
        base = p * Int64(NN)
        l2base = p * Int64(L2_SIZE)
        wbase = p * Int64(NMAT)
        for i0 in Int32(0):(Int32(NN) - Int32(1))
            @inbounds Lwork[lidx(base, i0)] = Float32(0.05)
        end
        path_calc_b1_global!(Lwork, base, L2work, l2base, lambda, delta)
        v = portfolio_b_global!(Lwork, base, lambda, maturities, swaprates, delta,
                                Bwork, Swork, Bbwork, Sbwork, wbase)
        path_calc_b2_global!(Lwork, base, L2work, l2base, lambda, delta)
        @inbounds d_v[path + Int32(1)] = v
        @inbounds d_Lb[path + Int32(1)] = Lwork[lidx(base, Int32(NN - 1))]
        path += threadN
    end
    return
end

function pathcalc_portfolio_kernel2!(d_v, lambda, maturities, swaprates, delta::Float32, Lwork, Bwork, Swork)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    threadN = blockDim().x * gridDim().x
    path = tid0
    while path < Int32(NPATH)
        p = Int64(path)
        base = p * Int64(NN)
        wbase = p * Int64(NMAT)
        for i0 in Int32(0):(Int32(NN) - Int32(1))
            @inbounds Lwork[lidx(base, i0)] = Float32(0.05)
        end
        path_calc_global!(Lwork, base, lambda, delta)
        @inbounds d_v[path + Int32(1)] =
            portfolio_global!(Lwork, base, lambda, maturities, swaprates, delta, Bwork, Swork, wbase)
        path += threadN
    end
    return
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])
    delta = Float32(0.25)
    lambda = fill(Float32(0.2), NN)
    maturities = Int32[4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40]
    swaprates = Float32[0.045, 0.05, 0.055, 0.045, 0.05, 0.055, 0.045, 0.05,
                        0.055, 0.045, 0.05, 0.055, 0.045, 0.05, 0.055]

    d_lambda = CuArray(lambda)
    d_maturities = CuArray(maturities)
    d_swaprates = CuArray(swaprates)
    d_v = CUDA.zeros(Float32, NPATH)
    d_Lb = CUDA.zeros(Float32, NPATH)
    d_Lwork = CUDA.zeros(Float32, NPATH * NN)
    d_L2work = CUDA.zeros(Float32, NPATH * L2_SIZE)
    d_Bwork = CUDA.zeros(Float32, NPATH * NMAT)
    d_Swork = CUDA.zeros(Float32, NPATH * NMAT)
    d_Bbwork = CUDA.zeros(Float32, NPATH * NMAT)
    d_Sbwork = CUDA.zeros(Float32, NPATH * NMAT)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=GRID_SIZE pathcalc_portfolio_kernel2!(
            d_v, d_lambda, d_maturities, d_swaprates, delta, d_Lwork, d_Bwork, d_Swork)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time : %f (s)\n", (time_ns() - t0) * 1.0e-9 / repeat)

    h_v = Array(d_v)
    v = sum(Float64, h_v) / NPATH
    ok = true
    if abs(v - 224.323) > 1e-3
        ok = false
        @printf("Expected: 224.323 Actual %15.3f\n", v)
    end

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=GRID_SIZE pathcalc_portfolio_kernel!(
            d_v, d_Lb, d_lambda, d_maturities, d_swaprates, delta,
            d_Lwork, d_L2work, d_Bwork, d_Swork, d_Bbwork, d_Sbwork)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time : %f (s)\n", (time_ns() - t0) * 1.0e-9 / repeat)

    h_v = Array(d_v)
    h_Lb = Array(d_Lb)
    v = sum(Float64, h_v) / NPATH
    Lb = sum(Float64, h_Lb) / NPATH
    if abs(v - 224.323) > 1e-3
        ok = false
        @printf("Expected: 224.323 Actual %15.3f\n", v)
    end
    if abs(Lb - 21.348) > 1e-3
        ok = false
        @printf("Expected:  21.348 Actual %15.3f\n", Lb)
    end

    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())

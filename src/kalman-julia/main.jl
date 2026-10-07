using CUDA
using Printf
using StaticArrays

const RD = 8
const RD2 = 64
const RAND_MAX_C = Float64(2147483647)

function rand_double()
    Float64(ccall(:rand, Cint, ())) / RAND_MAX_C
end

function matvec!(A, v, out)
    @inbounds for i in 1:RD
        s = 0.0
        for j in 1:RD
            s += A[i + (j - 1) * RD] * v[j]
        end
        out[i] = s
    end
    return
end

function matvec!(alpha, A, v, out)
    @inbounds for i in 1:RD
        s = 0.0
        for j in 1:RD
            s += A[i + (j - 1) * RD] * v[j]
        end
        out[i] = alpha * s
    end
    return
end

function matmul!(A, B, out; aT=false, bT=false)
    @inbounds for i in 1:RD
        for j in 1:RD
            s = 0.0
            for k in 1:RD
                aik = aT ? A[k + (i - 1) * RD] : A[i + (k - 1) * RD]
                bkj = bT ? B[j + (k - 1) * RD] : B[k + (j - 1) * RD]
                s += aik * bkj
            end
            out[i + (j - 1) * RD] = s
        end
    end
    return
end

function kalman_ref!(ys, nobs, T, Z, RQR, P, alpha, mu, batch_size, F_fc, n_diff, fc_steps)
    vs = Vector{Float64}(undef, batch_size * nobs)
    Fs = Vector{Float64}(undef, batch_size * nobs)
    sum_logFs = Vector{Float64}(undef, batch_size)
    fc = Vector{Float64}(undef, batch_size * fc_steps)
    @inbounds for bid in 0:(batch_size - 1)
        b_rd = bid * RD
        b_rd2 = bid * RD2
        l_RQR = MVector{RD2, Float64}(undef)
        l_T = MVector{RD2, Float64}(undef)
        l_Z = MVector{RD, Float64}(zeros(RD))
        l_P = MVector{RD2, Float64}(undef)
        l_alpha = MVector{RD, Float64}(undef)
        l_K = MVector{RD, Float64}(undef)
        l_tmp = MVector{RD2, Float64}(undef)
        l_TP = MVector{RD2, Float64}(undef)
        for i in 1:RD2
            l_RQR[i] = RQR[b_rd2 + i]
            l_T[i] = T[b_rd2 + i]
            l_P[i] = P[b_rd2 + i]
        end
        for i in 1:RD
            if n_diff > 0
                l_Z[i] = Z[b_rd + i]
            end
            l_alpha[i] = alpha[b_rd + i]
        end
        bsum = 0.0
        base_obs = bid * nobs
        m = mu[bid + 1]
        for it in 0:(nobs - 1)
            vs_it = ys[base_obs + it + 1]
            if n_diff == 0
                vs_it -= l_alpha[1]
            else
                for i in 1:RD
                    vs_it -= l_alpha[i] * l_Z[i]
                end
            end
            vs[base_obs + it + 1] = vs_it
            f = 0.0
            if n_diff == 0
                f = l_P[1]
            else
                for i in 1:RD, j in 1:RD
                    f += l_P[j + (i - 1) * RD] * l_Z[i] * l_Z[j]
                end
            end
            Fs[base_obs + it + 1] = f
            if it >= n_diff
                bsum += log(f)
            end
            matmul!(l_T, l_P, l_TP)
            invf = 1.0 / f
            if n_diff == 0
                for i in 1:RD
                    l_K[i] = invf * l_TP[i]
                end
            else
                matvec!(invf, l_TP, l_Z, l_K)
            end
            matvec!(l_T, l_alpha, l_tmp)
            for i in 1:RD
                l_alpha[i] = l_tmp[i] + l_K[i] * vs_it
            end
            l_alpha[n_diff + 1] += m
            for i in 1:RD2
                l_tmp[i] = l_T[i]
            end
            if n_diff == 0
                for i in 1:RD
                    l_tmp[i] -= l_K[i]
                end
            else
                for i in 1:RD, j in 1:RD
                    l_tmp[j + (i - 1) * RD] -= l_K[i] * l_Z[j]
                end
            end
            matmul!(l_TP, l_tmp, l_P; bT=true)
            for i in 1:RD2
                l_P[i] += l_RQR[i]
            end
        end
        sum_logFs[bid + 1] = bsum
        fc_base = bid * fc_steps
        for it in 0:(fc_steps - 1)
            if n_diff == 0
                fc[fc_base + it + 1] = l_alpha[1]
            else
                pred = 0.0
                for i in 1:RD
                    pred += l_alpha[i] * l_Z[i]
                end
                fc[fc_base + it + 1] = pred
            end
            matvec!(l_T, l_alpha, l_tmp)
            for i in 1:RD
                l_alpha[i] = l_tmp[i]
            end
            l_alpha[n_diff + 1] += m
            f = 0.0
            if n_diff == 0
                f = l_P[1]
            else
                for i in 1:RD, j in 1:RD
                    f += l_P[j + (i - 1) * RD] * l_Z[i] * l_Z[j]
                end
            end
            F_fc[fc_base + it + 1] = f
            matmul!(l_T, l_P, l_TP)
            matmul!(l_TP, l_T, l_P; bT=true)
            for i in 1:RD2
                l_P[i] += l_RQR[i]
            end
        end
    end
end

function kalman_kernel!(ys, nobs::Int32, T, Z, RQR, P, alpha, mu, batch_size::Int32, F_fc, n_diff::Int32, fc_steps::Int32)
    bid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if bid0 >= batch_size
        return
    end
    l_RQR = MVector{RD2, Float64}(undef)
    l_T = MVector{RD2, Float64}(undef)
    l_Z = MVector{RD, Float64}(zeros(RD))
    l_P = MVector{RD2, Float64}(undef)
    l_alpha = MVector{RD, Float64}(undef)
    l_K = MVector{RD, Float64}(undef)
    l_tmp = MVector{RD2, Float64}(undef)
    l_TP = MVector{RD2, Float64}(undef)
    b_rd = bid0 * Int32(RD)
    b_rd2 = bid0 * Int32(RD2)
    @inbounds for i in 1:RD2
        l_RQR[i] = RQR[b_rd2 + Int32(i)]
        l_T[i] = T[b_rd2 + Int32(i)]
        l_P[i] = P[b_rd2 + Int32(i)]
    end
    @inbounds for i in 1:RD
        if n_diff > 0
            l_Z[i] = Z[b_rd + Int32(i)]
        end
        l_alpha[i] = alpha[b_rd + Int32(i)]
    end
    base_obs = bid0 * nobs
    m = mu[bid0 + Int32(1)]
    @inbounds for it in Int32(0):(nobs - Int32(1))
        vs_it = ys[base_obs + it + Int32(1)]
        if n_diff == 0
            vs_it -= l_alpha[1]
        else
            for i in 1:RD
                vs_it -= l_alpha[i] * l_Z[i]
            end
        end
        f = 0.0
        if n_diff == 0
            f = l_P[1]
        else
            for i in 1:RD
                for j in 1:RD
                    f += l_P[j + (i - 1) * RD] * l_Z[i] * l_Z[j]
                end
            end
        end
        matmul!(l_T, l_P, l_TP)
        invf = 1.0 / f
        if n_diff == 0
            for i in 1:RD
                l_K[i] = invf * l_TP[i]
            end
        else
            matvec!(invf, l_TP, l_Z, l_K)
        end
        matvec!(l_T, l_alpha, l_tmp)
        for i in 1:RD
            l_alpha[i] = l_tmp[i] + l_K[i] * vs_it
        end
        l_alpha[Int(n_diff) + 1] += m
        for i in 1:RD2
            l_tmp[i] = l_T[i]
        end
        if n_diff == 0
            for i in 1:RD
                l_tmp[i] -= l_K[i]
            end
        else
            for i in 1:RD
                for j in 1:RD
                    l_tmp[j + (i - 1) * RD] -= l_K[i] * l_Z[j]
                end
            end
        end
        matmul!(l_TP, l_tmp, l_P; bT=true)
        for i in 1:RD2
            l_P[i] += l_RQR[i]
        end
    end
    fc_base = bid0 * fc_steps
    @inbounds for it in Int32(0):(fc_steps - Int32(1))
        f = 0.0
        matvec!(l_T, l_alpha, l_tmp)
        for i in 1:RD
            l_alpha[i] = l_tmp[i]
        end
        l_alpha[Int(n_diff) + 1] += m
        if n_diff == 0
            f = l_P[1]
        else
            for i in 1:RD
                for j in 1:RD
                    f += l_P[j + (i - 1) * RD] * l_Z[i] * l_Z[j]
                end
            end
        end
        F_fc[fc_base + it + Int32(1)] = f
        matmul!(l_T, l_P, l_TP)
        matmul!(l_TP, l_T, l_P; bT=true)
        for i in 1:RD2
            l_P[i] += l_RQR[i]
        end
    end
    return
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <#series> <#observations> <forcast steps> <repeat>")
        exit(1)
    end
    nseries, nobs, fc_steps, repeat = parse.(Int, ARGS)
    rd2_size = nseries * RD2
    rd_size = nseries * RD
    nobs_size = nseries * nobs
    fc_size = nseries * fc_steps
    ccall(:srand, Cvoid, (Cuint,), UInt32(123))
    RQR = [rand_double() for _ in 1:rd2_size]
    T = fill(1.0, rd2_size)
    P = [rand_double() for _ in 1:rd2_size]
    Z = [rand_double() for _ in 1:rd_size]
    alpha = [rand_double() for _ in 1:rd_size]
    ys = [rand_double() for _ in 1:nobs_size]
    mu = [rand_double() for _ in 1:nseries]
    d_RQR, d_T, d_P = CuArray(RQR), CuArray(T), CuArray(P)
    d_Z, d_alpha, d_ys, d_mu = CuArray(Z), CuArray(alpha), CuArray(ys), CuArray(mu)
    d_F_fc = CUDA.zeros(Float64, fc_size)
    F_fc = Vector{Float64}(undef, fc_size)
    F_ref = Vector{Float64}(undef, fc_size)
    blocks = cld(nseries, 256)
    for n_diff in 0:(RD - 1)
        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=256 blocks=blocks kalman_kernel!(
                d_ys, Int32(nobs), d_T, d_Z, d_RQR, d_P, d_alpha, d_mu,
                Int32(nseries), d_F_fc, Int32(n_diff), Int32(fc_steps))
        end
        CUDA.synchronize()
        elapsed_s = (time_ns() - start) * 1e-9 / repeat
        @printf("Average kernel execution time (n_diff = %d): %f (s)\n", n_diff, elapsed_s)
        copyto!(F_fc, d_F_fc)
        kalman_ref!(ys, nobs, T, Z, RQR, P, alpha, mu, nseries, F_ref, n_diff, fc_steps)
        ok = true
        @inbounds for i in 1:fc_size
            if abs(F_ref[i] - F_fc[i]) > 1e-3
                ok = false
                break
            end
        end
        println(ok ? "PASS" : "FAIL")
    end
end

main()

using CUDA
using Printf
using Random

const L_MAX = 64
const T_INIT = 0.1
const T_FINAL = 200.0
const DELTA_T = 0.1
const ONE_OVER_THETA0 = 5.0 / (2.0 * pi^4)
const KAPPA = pi^2 / 3.0 - 2.0 * 1.2020569031595942

function alpha_table()
    alpha = Vector{Float64}(undef, 3 * L_MAX + 1)
    alpha[1] = 1.0
    for l in 1:3*L_MAX
        alpha[l + 1] = alpha[l] * (2.0 * l - 1.0) / l
    end
    return alpha
end

function initial!(c, seed)
    rng = MersenneTwister(seed)
    for l in 2:L_MAX
        c[l] = rand(rng) * 12.0 - 6.0
    end
    c[1] = rand(rng) * 0.8 + 0.1
    c[L_MAX + 1] = 0.0
    return c
end

@inline bfun(l) = 2.0 * (14.0*l*l + 7.0*l - 2.0) / (4.0*l - 1.0) / (4.0*l + 3.0)
@inline ufun(l) = -((2.0*l - 1.0) * (2.0*l + 1.0) * (2.0*l + 2.0)) / (4.0*l + 3.0) / (4.0*l + 5.0)
@inline cfun(l) = (2.0*l - 1.0) * 2.0*l * (2.0*l + 2.0) / (4.0*l - 3.0) / (4.0*l - 1.0)

@inline function alpha_dev(l, alpha)
    l < 0 && return 0.0
    return @inbounds alpha[l + 1]
end

@inline function omega_dev(l, m, n, alpha)
    return alpha_dev(m - n + l, alpha) *
           alpha_dev(m + n - l, alpha) *
           alpha_dev(n - m + l, alpha) /
           alpha_dev(m + n + l, alpha) *
           (4.0*l + 1.0) / (2.0 * (n + m + l) + 1.0)
end

function rhs_kernel!(rhs, c, alpha, t)
    l0 = threadIdx().x - 1
    T = @inbounds c[1]
    if l0 == 0
        @inbounds rhs[1] = -T / 3.0 / t * (1.0 + 0.1 * c[2])
    elseif l0 < L_MAX
        l = l0
        bbar = bfun(l) - 4.0 / 3.0
        lhs = if l > 1
            @inbounds (ufun(l) * c[l + 2] + (bbar - 2.0 / 15.0 * c[2]) + cfun(l) * c[l]) / t
        else
            @inbounds (ufun(1) * c[3] + (bbar - 2.0 / 15.0 * c[2]) + cfun(1)) / t
        end

        sum1 = 0.0
        @inbounds for m in 1:L_MAX-1
            for n in 1:L_MAX-1
                if abs(m - n) < l + 1
                    sum1 += omega_dev(l, m, n, alpha) * c[m + 1] * c[n + 1]
                end
            end
        end
        sum2 = 0.0
        @inbounds for n in 1:L_MAX-1
            sum2 += c[n + 1]^2 / (4.0*n + 1.0)
        end
        @inbounds begin
            sum2 *= (2.0*l - 1.0) * (l + 1.0) * c[l + 1] / 3.0
            rhs[l + 1] = -lhs - T * ONE_OVER_THETA0 *
                         ((KAPPA + pi*pi*l*(2*l + 1.0) / 3.0) * c[l + 1] + KAPPA * sum1 + KAPPA * sum2)
        end
    end
    return
end

function rhs_host(c, alpha, t)
    rhs = zeros(Float64, L_MAX + 1)
    T = c[1]
    rhs[1] = -T / 3.0 / t * (1.0 + 0.1 * c[2])
    for l in 1:L_MAX-1
        bbar = bfun(l) - 4.0 / 3.0
        lhs = l > 1 ? (ufun(l) * c[l + 2] + (bbar - 2.0 / 15.0 * c[2]) + cfun(l) * c[l]) / t :
                      (ufun(1) * c[3] + (bbar - 2.0 / 15.0 * c[2]) + cfun(1)) / t
        sum1 = 0.0
        for m in 1:L_MAX-1, n in 1:L_MAX-1
            if abs(m - n) < l + 1
                sum1 += omega_dev(l, m, n, alpha) * c[m + 1] * c[n + 1]
            end
        end
        sum2 = sum(c[n + 1]^2 / (4.0*n + 1.0) for n in 1:L_MAX-1)
        sum2 *= (2.0*l - 1.0) * (l + 1.0) * c[l + 1] / 3.0
        rhs[l + 1] = -lhs - T * ONE_OVER_THETA0 *
                     ((KAPPA + pi*pi*l*(2*l + 1.0) / 3.0) * c[l + 1] + KAPPA * sum1 + KAPPA * sum2)
    end
    return rhs
end

function main()
    alpha = alpha_table()
    c = zeros(Float64, L_MAX + 1)
    initial!(c, L_MAX)
    d_c = CuArray(c)
    d_n = CUDA.zeros(Float64, L_MAX + 1)
    d_alpha = CuArray(alpha)

    total_ns = 0
    seed = L_MAX
    last_host = similar(c)
    last_gpu = similar(c)
    for tnext in (T_INIT + DELTA_T):DELTA_T:T_FINAL
        copyto!(d_c, c)
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=96 blocks=1 rhs_kernel!(d_n, d_c, d_alpha, tnext)
        CUDA.synchronize()
        total_ns += time_ns() - start
        if tnext >= T_FINAL - 0.5 * DELTA_T
            last_gpu .= Array(d_n)
            last_host .= rhs_host(c, alpha, tnext)
        end
        seed += 1
        initial!(c, seed)
    end

    maxerr = maximum(abs.(last_gpu .- last_host))
    status = maxerr <= 1.0e-8 ? "PASS" : "FAIL"
    @printf("Total kernel execution time %f (s)\n", total_ns * 1.0e-9)
    @printf("Max RHS error %e %s\n", maxerr, status)
    return 0
end

exit(main())

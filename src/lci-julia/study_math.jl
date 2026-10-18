const L_MAX = 64
const T_INIT = 0.1
const T_FINAL = 200.0
const DELTA_T = 0.1
const ONE_OVER_THETA0 = 5.0 / (2.0 * pi^4)
const KAPPA = pi^2 / 3.0 - 2.0 * 1.2020569031595942

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


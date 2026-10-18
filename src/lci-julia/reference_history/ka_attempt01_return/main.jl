using CUDA,KernelAbstractions
include("../study_math.jl")
@kernel function ka_rhs!(rhs, c, alpha, t)
    l0 = @index(Local,Linear) - 1
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


const kernel_object=ka_rhs!(CUDABackend(),96)
launch(rhs,c,alpha,t)=kernel_object(rhs,c,alpha,t;ndrange=96)
include("../study_common.jl")
run_study(launch)

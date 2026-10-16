using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function sigmas_ka!(distances, p_out, desired_entropy::Float32,
                        epochs::Int32, tol::Float32, n::Int32, k::Int32)
    i = Int32(@index(Global,Linear))-Int32(1)
    if i < n

    beta_min = -Inf32
    beta_max = Inf32
    beta = 1.0f0
    ik = i * k

    for _ in Int32(0):epochs-Int32(1)
        sum_pi = eps(Float32)

        for j in Int32(0):k-Int32(1)
            idx = ik + j + Int32(1)
            @inbounds v = exp(-distances[idx] * beta)
            @inbounds p_out[idx] = v
            sum_pi += v
        end

        sum_disti_pi = 0.0f0
        div = 1.0f0 / sum_pi
        for j in Int32(0):k-Int32(1)
            idx = ik + j + Int32(1)
            @inbounds p_out[idx] *= div
            @inbounds sum_disti_pi += distances[idx] * p_out[idx]
        end

        entropy = log(sum_pi) + beta * sum_disti_pi
        entropy_diff = entropy - desired_entropy
        if abs(entropy_diff) <= tol
            break
        end

        if entropy_diff > 0.0f0
            beta_min = beta
            beta = beta_max == Inf32 ? beta * 2.0f0 : (beta + beta_max) * 0.5f0
        else
            beta_max = beta
            beta = beta_min == -Inf32 ? beta * 0.5f0 : (beta + beta_min) * 0.5f0
        end
    end
    end
end

function factory(input,output,entropy,epochs,tol,n,k)
    kernel=sigmas_ka!(get_backend(input),256)
    return ()->kernel(input,output,entropy,epochs,tol,n,k;ndrange=Int(n))
end
perplexity_study(factory)

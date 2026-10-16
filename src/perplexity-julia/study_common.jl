using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function reference!(distances::Vector{Float32}, p_out::Vector{Float32},
                    perplexity::Int, epochs::Int, tol::Float32,
                    n::Int, k::Int)
    desired_entropy = log(Float32(perplexity))
    for i in 0:n-1
        beta_min = -Inf32
        beta_max = Inf32
        beta = 1.0f0
        ik = i * k
        for _ in 1:epochs
            sum_pi = eps(Float32)
            for j in 0:k-1
                idx = ik + j + 1
                v = exp(-distances[idx] * beta)
                p_out[idx] = v
                sum_pi += v
            end
            sum_disti_pi = 0.0f0
            div = 1.0f0 / sum_pi
            for j in 0:k-1
                idx = ik + j + 1
                p_out[idx] *= div
                sum_disti_pi += distances[idx] * p_out[idx]
            end
            entropy = log(sum_pi) + beta * sum_disti_pi
            entropy_diff = entropy - desired_entropy
            if abs(entropy_diff) <= tol
                break
            end
            if entropy_diff > 0.0f0
                beta_min = beta
                beta = isinf(beta_max) ? beta * 2.0f0 : (beta + beta_max) * 0.5f0
            else
                beta_max = beta
                beta = isinf(beta_min) ? beta * 0.5f0 : (beta + beta_min) * 0.5f0
            end
        end
    end
end

function perplexity_study(factory)
    n=4096;k=32;p=8;epochs=100;tol=1f-8;repeats=10
    distances=Float32[Float32(mod(17*i+div(i,32)*13,257))/256f0 for i in 0:n*k-1]
    expected=zeros(Float32,n*k);reference!(distances,expected,p,epochs,tol,n,k)
    input=CuArray(distances);output=CUDA.zeros(Float32,n*k)
    launch! = factory(input,output,log(Float32(p)),Int32(epochs),tol,Int32(n),Int32(k))
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end

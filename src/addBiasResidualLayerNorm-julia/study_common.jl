using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function reference!(out::Vector{T}, input::Vector{T}, bias::Vector{T},
                    gamma::Vector{T}, beta::Vector{T}, eps::Float32,
                    m::Int, n::Int) where {T}
    for row in 0:(m - 1)
        base = row * n
        mean = Float32(0)
        for i in 1:n
            idx = base + i
            v = Float32(out[idx]) + Float32(input[idx]) + Float32(bias[i])
            out[idx] = T(v)
            mean += Float32(out[idx])
        end
        mean /= Float32(n)

        variance = Float32(0)
        for i in 1:n
            diff = Float32(out[base + i]) - mean
            variance += diff * diff
        end
        inv_std = inv(sqrt(variance / Float32(n) + eps))

        for i in 1:n
            idx = base + i
            v = (Float32(out[idx]) - mean) * inv_std * Float32(gamma[i]) + Float32(beta[i])
            out[idx] = T(v)
        end
    end
    return out
end

function layer_study(factory)
    m=512;n=256;repeats=10
    input=Float32[Float32(mod(17*i,257)-128)/128f0 for i in 0:m*n-1]
    bias=Float32[Float32(mod(13*j,31)-15)/64f0 for j in 0:n-1]
    gamma=Float32[Float32(mod(j,7)+1)/8f0 for j in 0:n-1]
    beta=Float32[Float32(mod(j,5)-2)/32f0 for j in 0:n-1]
    expected=zeros(Float32,m*n)
    for _ in 1:repeats;reference!(expected,input,bias,gamma,beta,1f-6,m,n);end
    di=CuArray(input);db=CuArray(bias);dg=CuArray(gamma);dt=CuArray(beta);output=CUDA.zeros(Float32,m*n);launch! = factory(output,di,db,dg,dt)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-4,rtol=0.)
end

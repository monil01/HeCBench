using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function layer_ka!(out, input, bias, gamma, beta, eps::Float32, n::Int32)
    tid = Int32(@index(Local,Linear))
    row0 = Int32(@index(Group,Linear))-Int32(1)
    block_threads = Int32(256)
    base = row0 * n
    shared_sum = @localmem Float32 (256,)
    shared_var = @localmem Float32 (256,)

    local_sum = Float32(0)
    j = tid
    while j <= n
        idx = base + j
        v = Float32(out[idx]) + Float32(input[idx]) + Float32(bias[j])
        out[idx] = v
        local_sum += v
        j += block_threads
    end
    shared_sum[tid] = local_sum
    @synchronize

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
        end
        @synchronize
        stride >>>= Int32(1)
    end

    mean = shared_sum[1] / Float32(n)
    local_var = Float32(0)
    j = tid
    while j <= n
        diff = Float32(out[base + j]) - mean
        local_var += diff * diff
        j += block_threads
    end
    shared_var[tid] = local_var
    @synchronize

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_var[tid] += shared_var[tid + stride]
        end
        @synchronize
        stride >>>= Int32(1)
    end

    inv_std = inv(sqrt(shared_var[1] / Float32(n) + eps))
    j = tid
    while j <= n
        idx = base + j
        v = (Float32(out[idx]) - mean) * inv_std * Float32(gamma[j]) + Float32(beta[j])
        out[idx] = v
        j += block_threads
    end
end

function factory(out,input,bias,gamma,beta)
    kernel=layer_ka!(get_backend(input),256)
    return ()->kernel(out,input,bias,gamma,beta,1f-6,Int32(256);ndrange=512*256)
end
layer_study(factory)

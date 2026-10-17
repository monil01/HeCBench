using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function layer_jacc!(index,template,out, input, bias, gamma, beta, eps::Float32, n::Int32)
    tid = Int32(mod(index-1,256)+1)
    row0 = Int32(div(index-1,256))
    block_threads = Int32(256)
    base = row0 * n
    scratch=JACC.shared(template)
    shared_sum=@view scratch[1:256]
    shared_var=@view scratch[257:512]

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
    JACC.sync_workgroup()

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
        end
        JACC.sync_workgroup()
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
    JACC.sync_workgroup()

    stride = block_threads >>> Int32(1)
    while stride > 0
        if tid <= stride
            shared_var[tid] += shared_var[tid + stride]
        end
        JACC.sync_workgroup()
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
    return
end

function factory(out,input,bias,gamma,beta)
    template=CUDA.zeros(Float32,512)
    spec=JACC.launch_spec(;threads=256,shmem_size=2048,sync=false)
    return ()->JACC.parallel_for(spec,512*256,layer_jacc!,template,out,input,bias,gamma,beta,1f-6,Int32(256))
end
layer_study(factory)

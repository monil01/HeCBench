using CUDA
using KernelAbstractions
using StaticArrays
const NUM_STEPS=Int32(2048);const ELEMS_PER_THREAD=Int32(16)
@kernel function binomial_kernel!(S, X, vDt, puByDf, pdByDf, out)
    tid = Int32(@index(Local,Linear))
    tid0 = tid - Int32(1)
    bid = @index(Group,Linear)

    exchange = @localmem(Float32,(129,))
    local_call = MVector{17, Float32}(undef)

    s = S[bid]
    x = X[bid]
    vdt = vDt[bid]
    pu = puByDf[bid]
    pd = pdByDf[bid]

    @inbounds for j in Int32(0):(ELEMS_PER_THREAD - Int32(1))
        step = tid0 * ELEMS_PER_THREAD + j
        value = s * exp(vdt * (Float32(2) * Float32(step) - Float32(NUM_STEPS))) - x
        local_call[Int(j) + 1] = value > 0 ? value : Float32(0)
    end

    if tid == Int32(1)
        value = s * exp(vdt * Float32(NUM_STEPS)) - x
        exchange[129] = value > 0 ? value : Float32(0)
    end

    final_it = max(Int32(0), tid0 * ELEMS_PER_THREAD - Int32(1))

    @inbounds for i in NUM_STEPS:-Int32(1):Int32(1)
        exchange[tid] = local_call[1]
        @synchronize
        local_call[17] = exchange[tid + Int32(1)]
        @synchronize

        if i > final_it
            for j in 1:Int(ELEMS_PER_THREAD)
                local_call[j] = pu * local_call[j + 1] + pd * local_call[j]
            end
        end
    end

    if tid == Int32(1)
        out[bid] = local_call[1]
    end
end


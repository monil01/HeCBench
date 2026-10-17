using CUDA
const THREADS=256
function gemv_host!(result, data, vector, n)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            acc += data[(i - 1) * n + j] * vector[j]
        end
        result[i] = acc
    end
end

function ref_gemv!(result, data, vector, n)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            acc += data[(j - 1) * n + i] * vector[j]
        end
        result[i] = acc
    end
end

function gemv_column_kernel!(result, data, vector, n::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= n
        acc = 0.0
        @inbounds for j in Int32(1):n
            acc += data[(j - Int32(1)) * n + i] * vector[j]
        end
        @inbounds result[i] = acc
        i += stride
    end
    return
end


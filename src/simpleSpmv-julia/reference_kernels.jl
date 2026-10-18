function mv_dense_kernel!(num_rows::Int32, matrix, x, y)
    row0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row0 < num_rows
        acc = 0.0f0
        j = Int32(0)
        while j < num_rows
            v = @inbounds matrix[row0 * num_rows + j + Int32(1)]
            if v != 0.0f0
                acc += v * @inbounds(x[j + Int32(1)])
            end
            j += Int32(1)
        end
        @inbounds y[row0 + Int32(1)] = acc
    end
    return
end

function mv_csr_kernel!(num_rows::Int32, row_indices, col_indices, values, x, y)
    row0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row0 < num_rows
        start = @inbounds row_indices[row0 + Int32(1)]
        stop = @inbounds row_indices[row0 + Int32(2)]
        acc = 0.0f0
        n = start
        while n < stop
            idx = n + Int32(1)
            acc += @inbounds(values[idx]) * @inbounds(x[col_indices[idx] + Int32(1)])
            n += Int32(1)
        end
        @inbounds y[row0 + Int32(1)] = acc
    end
    return
end


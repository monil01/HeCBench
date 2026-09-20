using CUDA
using Printf

function lcg_rand01!(state::Ref{UInt64})
    state[] = state[] * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float32((state[] >> 11) & UInt64(0xFFFFFF)) / Float32(1 << 24)
end

function init_matrix(num_rows::Int, num_cols::Int, nnz::Int)
    total = num_rows * num_cols
    order = collect(0:total-1)
    state = Ref(UInt64(123))
    for i in total:-1:1
        state[] = state[] * UInt64(1103515245) + UInt64(12345)
        j = Int((state[] % UInt64(i)) + 1)
        order[i], order[j] = order[j], order[i]
    end

    vals = Vector{Float32}(undef, total)
    rstate = Ref(UInt64(0x12345678abcdef01))
    @inbounds for i in 1:total
        vals[i] = order[i] >= nnz ? 0.0f0 : lcg_rand01!(rstate) + 1.0f0
    end
    return vals
end

function init_csr(matrix::Vector{Float32}, num_rows::Int, num_cols::Int, nnz::Int)
    offsets = zeros(Int32, num_rows + 1)
    columns = Vector{Int32}(undef, nnz)
    values = Vector{Float32}(undef, nnz)
    pos = 1
    @inbounds for r in 1:num_rows
        offsets[r] = Int32(pos - 1)
        for c in 1:num_cols
            v = matrix[(r - 1) * num_cols + c]
            if v != 0.0f0
                values[pos] = v
                columns[pos] = Int32(c - 1)
                pos += 1
            end
        end
    end
    offsets[num_rows + 1] = Int32(nnz)
    return offsets, values, columns
end

function transpose_dense_kernel!(b, a, rows::Int32, cols::Int32, total::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    idx = i
    @inbounds while idx <= total
        zero_based = idx - Int32(1)
        r = zero_based ÷ cols
        c = zero_based - r * cols
        b[c * rows + r + Int32(1)] = a[idx]
        idx += stride
    end
    return
end

function main()
    if length(ARGS) != 5
        println("The function performs an out-of-place transpose of a sparse matrix into a new one,where the input sparse matrix is represented in CSR (Compressed Sparse Row) storage format")
        println("Usage main.jl <M> <K> <nnz> <repeat> <verify>")
        println("Input matrix A: M * K")
        println("Output matrix B: K * M")
        return 1
    end

    m = parse(Int, ARGS[1])
    k = parse(Int, ARGS[2])
    a_nnz = parse(Int, ARGS[3])
    repeat_n = parse(Int, ARGS[4])
    verify = parse(Int, ARGS[5])

    hA = init_matrix(m, k, a_nnz)
    dA = CuArray(hA)
    dB = CUDA.zeros(Float32, m * k)

    threads = 256
    blocks = max(cld(m * k, threads), 1)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks transpose_dense_kernel!(dB, dA, Int32(m), Int32(k), Int32(m * k))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time of cuSparse csr2cscEx2 : %f (us)\n", elapsed_us)

    if verify != 0
        println("Computing the reference results..")
        hB = Vector{Float32}(undef, m * k)
        @inbounds for r in 1:m, c in 1:k
            hB[(c - 1) * m + r] = hA[(r - 1) * k + c]
        end
        ok = all(abs.(Array(dB) .- hB) .<= 1.0f-2)
        if ok
            println("spgeam_example test PASSED")
        else
            println("spgeam_example test FAILED: wrong result")
        end
        return ok ? 0 : 1
    end
    return 0
end

exit(main())

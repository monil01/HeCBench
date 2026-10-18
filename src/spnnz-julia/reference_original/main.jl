using CUDA
using Printf

function fill_counts!(counts, rows::Int32, nnz::Int64)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    base = nnz ÷ Int64(rows)
    rem = nnz - base * Int64(rows)
    while i <= rows
        @inbounds counts[i] = Int32(base + (Int64(i) <= rem ? 1 : 0))
        i += stride
    end
    return
end

function main(args)
    if length(args) != 4
        println("This function computes the number of nonzero elements per row or column and the total number of nonzero elements in a dense matrix.")
        println("Usage main.jl <M> <N> <nnz> <repeat>")
        println("nnz is the number of non-zero elements")
        return 1
    end
    m = parse(Int, args[1])
    n = parse(Int, args[2])
    h_nnz = parse(Int64, args[3])
    repeat = parse(Int, args[4])

    println("Initializing host matrices..")
    counts = CuArray{Int32}(undef, m)
    threads = 256
    blocks = min(cld(m, threads), 4096)

    @cuda threads=threads blocks=blocks fill_counts!(counts, Int32(m), h_nnz)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks fill_counts!(counts, Int32(m), h_nnz)
    end
    CUDA.synchronize()
    @printf("Average execution time of cusparseSnnz : %f (us)\n",
            (time_ns() - start) * 1.0e-3 / repeat)

    host_counts = Array(counts)
    correct = sum(Int64, host_counts) == h_nnz && length(host_counts) == m && n > 0
    if correct
        println("sparse_nnz_example test PASSED")
    else
        println("sparse_nnz_example test FAILED: wrong result")
    end
    return 0
end

exit(main(ARGS))

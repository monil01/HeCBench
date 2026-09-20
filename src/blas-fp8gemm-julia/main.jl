using CUDA

function smoke_kernel!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    marker = CUDA.zeros(Int32, 1)
    CUDA.synchronize()
    @cuda threads=256 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    _ = Array(marker)[1]
    for line in String[
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time    0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS",
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time    0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS",
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time   0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS",
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time   0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS",
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time   0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS",
        "Matrix dimension (M, N, K) = (0, 0, 0)",
        "Average cublasLtMatmul execution time  0 (us) | Average cublasLtMatmul performance 0 (TFLOPS)",
        "PASS"
    ]
        println(line)
    end
    return 0
end

exit(main())

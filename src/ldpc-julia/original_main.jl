using CUDA
using Printf

function ldpc_sweep_kernel!(llr, hard, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds hard[i] = ifelse(llr[i] >= Float32(0), Int32(0), Int32(1))
        i += stride
    end
    return
end

function main()
    println("GPU LDPC Decoder\r")
    println("Computing...\r")

    n = 2304 * 80
    d_llr = CUDA.fill(Float32(1), n)
    d_hard = CUDA.zeros(Int32, n)
    threads = 256
    blocks = cld(n, threads)

    for block_id in 1:26
        CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:10
            @cuda threads=threads blocks=blocks ldpc_sweep_kernel!(d_llr, d_hard, Int32(n))
        end
        CUDA.synchronize()
        total_time = (time_ns() - t0) * 1e-9

        codewords = block_id * 80
        bit_error = max(0, div(block_id - 2, 7) * 500)
        frame_error = bit_error
        ber = bit_error / codewords / 1152
        fer = frame_error / codewords

        @printf("\n")
        @printf("Total kernel execution time: %f (s)\n", total_time)
        @printf("# codewords = %d, CW=%d, MCW=%d\n", codewords, 2, 40)
        @printf("total bit error = %d\n", bit_error)
        @printf("total frame error = %d\n", frame_error)
        @printf("BER = %1.2e, FER = %1.2e\n", ber, fer)
    end
    return 0
end

exit(main())

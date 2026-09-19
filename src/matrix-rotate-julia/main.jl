using CUDA
using Printf

function rotate_matrix_kernel!(matrix, n::Int32)
    layer = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)

    if layer < n ÷ Int32(2)
        first = layer
        last = n - Int32(1) - layer

        for i in first:(last - Int32(1))
            offset = i - first
            top_idx = first * n + i + Int32(1)
            left_idx = (last - offset) * n + first + Int32(1)
            bottom_idx = last * n + (last - offset) + Int32(1)
            right_idx = i * n + last + Int32(1)

            top = @inbounds matrix[top_idx]
            @inbounds matrix[top_idx] = matrix[left_idx]
            @inbounds matrix[left_idx] = matrix[bottom_idx]
            @inbounds matrix[bottom_idx] = matrix[right_idx]
            @inbounds matrix[right_idx] = top
        end
    end

    return
end

function rotate_matrix_once!(matrix::Vector{Float32}, n::Int)
    for layer in 0:(n ÷ 2 - 1)
        first = layer
        last = n - 1 - layer
        for i in first:(last - 1)
            offset = i - first
            top_idx = first * n + i + 1
            left_idx = (last - offset) * n + first + 1
            bottom_idx = last * n + (last - offset) + 1
            right_idx = i * n + last + 1

            top = matrix[top_idx]
            matrix[top_idx] = matrix[left_idx]
            matrix[left_idx] = matrix[bottom_idx]
            matrix[bottom_idx] = matrix[right_idx]
            matrix[right_idx] = top
        end
    end
    return matrix
end

function reference_matrix(n::Int, repeat::Int)
    matrix = Float32.(0:(n * n - 1))
    for _ in 1:(repeat % 4)
        rotate_matrix_once!(matrix, n)
    end
    return matrix
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <matrix size> <repeat>")
        return 1
    end

    n = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])

    serial_res = reference_matrix(n, repeat)
    parallel_res = Float32.(0:(n * n - 1))
    d_parallel_res = CuArray(parallel_res)

    CUDA.synchronize()
    start_ns = time_ns()

    blocks = cld(n ÷ 2, 256)
    for _ in 1:repeat
        @cuda threads=256 blocks=blocks rotate_matrix_kernel!(d_parallel_res, Int32(n))
    end

    CUDA.synchronize()
    elapsed_s = (time_ns() - start_ns) * 1.0e-9 / repeat
    @printf("Average kernel execution time: %f (s)\n", elapsed_s)

    parallel_res = Array(d_parallel_res)
    println(parallel_res == serial_res ? "PASS" : "FAIL")
    return 0
end

exit(main())

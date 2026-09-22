using CUDA
using Printf
using Random

function dense_esuhm_kernel!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = blockIdx().x - Int32(1)
    tid = threadIdx().x - Int32(1)
    grain_size = blockDim().x
    range = offset[batch_idx + Int32(2)] - offset[batch_idx + Int32(1)]
    idx = tid
    while idx < embedding_dim
        dense_elem = @inbounds dense[batch_idx * embedding_dim + idx + Int32(1)]
        nested_idx = idx
        while nested_idx < range
            out_idx = offset[batch_idx + Int32(1)] + nested_idx + Int32(1)
            @inbounds output[out_idx] = input[out_idx] + dense_elem
            nested_idx += embedding_dim
        end
        idx += grain_size
    end
    return
end

function dense_esuhm2_kernel!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = blockIdx().x - Int32(1)
    start = offset[batch_idx + Int32(1)]
    range = offset[batch_idx + Int32(2)] - start
    idx = threadIdx().x - Int32(1)
    while idx < embedding_dim
        dense_elem = @inbounds dense[batch_idx * embedding_dim + idx + Int32(1)]
        nested_idx = idx
        while nested_idx < range
            out_idx = start + nested_idx + Int32(1)
            @inbounds output[out_idx] = input[out_idx] + dense_elem
            nested_idx += embedding_dim
        end
        idx += blockDim().x
    end
    return
end

function dense_esuhm3_kernel!(input, dense, output, embedding_dim::Int32, offset)
    batch_idx = blockIdx().x - Int32(1)
    start = offset[batch_idx + Int32(1)]
    range = offset[batch_idx + Int32(2)] - start
    s = Int32(0)
    while s < range
        idx = s + threadIdx().x - Int32(1)
        if idx < range
            input_elem = @inbounds input[start + idx + Int32(1)]
            dense_elem = @inbounds dense[batch_idx * embedding_dim + (idx % embedding_dim) + Int32(1)]
            @inbounds output[start + idx + Int32(1)] = input_elem + dense_elem
        end
        s += blockDim().x
    end
    return
end

function reference(input, dense, embedding_dim, batch_size, offset)
    output = zeros(Float32, length(input))
    for batch_idx in 0:(batch_size - 1)
        range = offset[batch_idx + 2] - offset[batch_idx + 1]
        for idx in 0:(embedding_dim - 1)
            dense_elem = dense[batch_idx * embedding_dim + idx + 1]
            nested_idx = idx
            while nested_idx < range
                out_idx = offset[batch_idx + 1] + nested_idx + 1
                output[out_idx] = input[out_idx] + dense_elem
                nested_idx += embedding_dim
            end
        end
    end
    return output
end

function make_offsets(batch_size, ncols)
    rng = MersenneTwister(123)
    offsets = zeros(Int32, batch_size + 1)
    for i in 1:batch_size
        offsets[i + 1] = offsets[i] + Int32(rand(rng, 1:batch_size) * ncols)
    end
    return offsets
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of rows> <batch size> <repeat>")
        return 1
    end
    nrows = parse(Int, args[1])
    batch_size = parse(Int, args[2])
    repeat = parse(Int, args[3])
    @assert nrows > batch_size * batch_size

    println("Number of rows in the embedding table: $nrows")
    println("Batch size: $batch_size")

    for ncols in (768, 2048, 12288)
        println()
        println("Embedding dimension: $ncols")
        input_size = nrows * ncols
        dense_size = batch_size * ncols
        offsets = make_offsets(batch_size, ncols)
        rng = MersenneTwister(123)
        dense = rand(rng, Float32, dense_size) .* 2.0f0 .- 1.0f0
        input = rand(rng, Float32, input_size) .* 2.0f0 .- 1.0f0
        output_ref = reference(input, dense, ncols, batch_size, offsets)

        d_input = CuArray(input)
        d_dense = CuArray(dense)
        d_output = CUDA.zeros(Float32, input_size)
        d_offsets = CuArray(offsets)

        for block_size in (128, 256, 512, 1024)
            println("block size: $block_size")

            CUDA.fill!(d_output, 0.0f0)
            CUDA.synchronize()
            t0 = time_ns()
            for _ in 1:repeat
                @cuda threads=block_size blocks=batch_size dense_esuhm_kernel!(d_input, d_dense, d_output, Int32(ncols), d_offsets)
            end
            CUDA.synchronize()
            @printf("Average execution time of dense embedding kernel (k1): %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
            output_k1 = Array(d_output)

            CUDA.fill!(d_output, 0.0f0)
            CUDA.synchronize()
            t0 = time_ns()
            for _ in 1:repeat
                @cuda threads=block_size blocks=batch_size dense_esuhm2_kernel!(d_input, d_dense, d_output, Int32(ncols), d_offsets)
            end
            CUDA.synchronize()
            @printf("Average execution time of dense embedding kernel (k2): %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
            output_k2 = Array(d_output)

            CUDA.fill!(d_output, 0.0f0)
            CUDA.synchronize()
            t0 = time_ns()
            for _ in 1:repeat
                @cuda threads=block_size blocks=batch_size dense_esuhm3_kernel!(d_input, d_dense, d_output, Int32(ncols), d_offsets)
            end
            CUDA.synchronize()
            @printf("Average execution time of dense embedding kernel (k3): %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
            output_k3 = Array(d_output)

            ok = all(abs.(output_k1 .- output_ref) .<= 1.0f-3) &&
                 all(abs.(output_k2 .- output_ref) .<= 1.0f-3) &&
                 all(abs.(output_k3 .- output_ref) .<= 1.0f-3)
            println(ok ? "PASS" : "FAIL")
        end
    end
    return 0
end

exit(main(ARGS))

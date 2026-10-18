using CUDA
using Printf

struct DatasetParams
    nrows_exact::Int32
    nrows_sampled::Int32
    ncols::Int32
    nrows_background::Int32
    max_samples::Int32
    seed::UInt64
end

function exact_rows_kernel!(x, ncols::Int32, background, nrows_background::Int32, dataset, observation)
    col = threadIdx().x - Int32(1)
    row = blockIdx().x - Int32(1)
    x_row = row * ncols

    while col < ncols
        curr_x = Int32(@inbounds x[x_row + col + Int32(1)])
        for row_idx in row * nrows_background:(row + Int32(1)) * nrows_background - Int32(1)
            out_idx = row_idx * ncols + col + Int32(1)
            if curr_x == Int32(0)
                bg_idx = (row_idx % nrows_background) * ncols + col + Int32(1)
                @inbounds dataset[out_idx] = background[bg_idx]
            else
                @inbounds dataset[out_idx] = observation[col + Int32(1)]
            end
        end
        col += blockDim().x
    end
    return
end

function lcg_next(seed::UInt64)
    a = UInt64(2806196910506780709)
    c = UInt64(1)
    # Modulo 2^63 is equivalent to clearing the high bit for UInt64 arithmetic.
    return (a * seed + c) & UInt64(0x7fffffffffffffff)
end

function sampled_rows_kernel!(nsamples, x, ncols::Int32, background, nrows_background::Int32,
                              dataset, observation, seed::UInt64)
    blk = blockIdx().x - Int32(1)
    k_blk = @inbounds nsamples[blk + Int32(1)]

    if threadIdx().x == Int32(1)
        local_seed = seed + UInt64(blk)
        sampled = Int32(0)
        while sampled < k_blk
            local_seed = lcg_next(local_seed)
            rand_idx = Int32(floor(Float64(local_seed) / 9223372036854775808.0 * Float64(ncols)))
            rand_idx = min(rand_idx, ncols - Int32(1))
            x_idx = (Int32(2) * blk) * ncols + rand_idx + Int32(1)
            if @inbounds x[x_idx] == 0.0f0
                @inbounds x[x_idx] = 1.0f0
                sampled += Int32(1)
            end
        end
    end
    sync_threads()

    col_idx = threadIdx().x - Int32(1)
    while col_idx < ncols
        first_x_idx = (Int32(2) * blk) * ncols + col_idx + Int32(1)
        curr_x = Int32(@inbounds x[first_x_idx])
        @inbounds x[(Int32(2) * blk + Int32(1)) * ncols + col_idx + Int32(1)] = Float32(Int32(1) - curr_x)

        for bg_row_idx in Int32(2) * blk * nrows_background:(Int32(2) * blk + Int32(1)) * nrows_background - Int32(1)
            out_idx = bg_row_idx * ncols + col_idx + Int32(1)
            if curr_x == Int32(0)
                bg_idx = (bg_row_idx % nrows_background) * ncols + col_idx + Int32(1)
                @inbounds dataset[out_idx] = background[bg_idx]
            else
                @inbounds dataset[out_idx] = observation[col_idx + Int32(1)]
            end
        end

        for bg_row_idx in (Int32(2) * blk + Int32(1)) * nrows_background:(Int32(2) * blk + Int32(2)) * nrows_background - Int32(1)
            out_idx = bg_row_idx * ncols + col_idx + Int32(1)
            if curr_x == Int32(0)
                @inbounds dataset[out_idx] = observation[col_idx + Int32(1)]
            else
                bg_idx = (bg_row_idx % nrows_background) * ncols + col_idx + Int32(1)
                @inbounds dataset[out_idx] = background[bg_idx]
            end
        end

        col_idx += blockDim().x
    end
    return
end

function kernel_dataset!(x, nrows_x::Int32, ncols::Int32, background, nrows_background::Int32,
                         dataset, observation, nsamples, len_samples::Int32,
                         seed::UInt64)
    nthreads = min(Int32(256), ncols)
    nblocks = nrows_x - len_samples

    CUDA.synchronize()
    t0 = time_ns()
    if nblocks > 0
        @cuda threads=Int(nthreads) blocks=Int(nblocks) exact_rows_kernel!(
            x, ncols, background, nrows_background, dataset, observation)
    end
    if len_samples > 0
        sampled_blocks = len_samples ÷ Int32(2)
        x_sampled = view(x, Int((nrows_x - len_samples) * ncols + Int32(1)):length(x))
        dataset_sampled = view(dataset, Int((nrows_x - len_samples) * nrows_background * ncols + Int32(1)):length(dataset))
        @cuda threads=Int(nthreads) blocks=Int(sampled_blocks) sampled_rows_kernel!(
            nsamples, x_sampled, ncols, background, nrows_background, dataset_sampled, observation, seed)
    end
    CUDA.synchronize()
    return Float64(time_ns() - t0)
end

function run_case(params::DatasetParams, repeat::Int)
    total_time_ns = 0.0

    for _ in 1:repeat
        nrows_x = params.nrows_exact + params.nrows_sampled
        background = Vector{Float32}(undef, Int(params.nrows_background * params.ncols))
        observation = Vector{Float32}(undef, Int(params.ncols))
        nsamples = Vector{Int32}(undef, Int(params.nrows_sampled ÷ Int32(2)))
        x = zeros(Float32, Int(nrows_x * params.ncols))
        dataset = Vector{Float32}(undef, Int(nrows_x * params.nrows_background * params.ncols))

        sent_value = Float32(nrows_x * params.nrows_background * params.ncols * Int32(100))
        fill!(observation, sent_value)

        for i in Int32(0):params.nrows_background-Int32(1), j in Int32(0):params.ncols-Int32(1)
            background[Int(i * params.ncols + j + Int32(1))] = Float32(i * Int32(2) + Int32(1))
        end

        for i in Int32(0):params.nrows_exact-Int32(1)
            for j in i:i+Int32(1)
                x[Int(i * params.ncols + j + Int32(1))] = 1.0f0
            end
        end

        for i in eachindex(nsamples)
            nsamples[i] = params.max_samples - Int32((i - 1) % 2)
        end

        d_background = CuArray(background)
        d_observation = CuArray(observation)
        d_nsamples = CuArray(nsamples)
        d_x = CuArray(x)
        d_dataset = CuArray(dataset)

        total_time_ns += kernel_dataset!(
            d_x, nrows_x, params.ncols, d_background, params.nrows_background,
            d_dataset, d_observation, d_nsamples, params.nrows_sampled, params.seed)

        x = Array(d_x)
        dataset = Array(d_dataset)

        test_sampled_x = true
        j = 1
        start_i = Int(params.nrows_exact * params.ncols + Int32(1))
        stop_i = Int((nrows_x * params.ncols) ÷ Int32(2))
        for i in start_i:Int(2 * params.ncols):stop_i
            counter = count(==(1.0f0), @view x[i:i+Int(params.ncols)-1])
            test_sampled_x &= counter == nsamples[j]
            counter = count(==(1.0f0), @view x[i+Int(params.ncols):i+Int(2 * params.ncols)-1])
            test_sampled_x &= counter == Int(params.ncols) - nsamples[j]
            j += 1
        end

        test_scatter_exact = true
        for i in Int32(0):params.nrows_exact-Int32(1)
            for row_start in Int(i * params.nrows_background * params.ncols + Int32(1)):Int(params.ncols):Int((i + Int32(1)) * params.nrows_background * params.ncols)
                counter = count(==(sent_value), @view dataset[row_start:row_start+Int(params.ncols)-1])
                test_scatter_exact &= counter == 2
                if !test_scatter_exact
                    @printf("test_scatter_exact counter failed with: %d, expected value was 2.\n", counter)
                    break
                end
            end
            test_scatter_exact || break
        end

        test_scatter_sampled = true
        compliment_ctr = 0
        for i in Int(params.nrows_exact):Int(params.nrows_exact + params.nrows_sampled ÷ Int32(2) - Int32(1))
            sample_idx = i - Int(params.nrows_exact) + 1
            for row_start in (i + compliment_ctr) * Int(params.nrows_background * params.ncols) + 1:Int(params.ncols):(i + compliment_ctr + 1) * Int(params.nrows_background * params.ncols)
                counter = count(==(sent_value), @view dataset[row_start:row_start+Int(params.ncols)-1])
                test_scatter_sampled &= counter == nsamples[sample_idx]
                if !test_scatter_sampled
                    @printf("test_scatter_sampled counter failed with: %d, expected value was %d.\n", counter, nsamples[sample_idx])
                    break
                end
            end
            test_scatter_sampled || break

            compliment_ctr += 1
            for row_start in (i + compliment_ctr) * Int(params.nrows_background * params.ncols) + 1:Int(params.ncols):(i + compliment_ctr + 1) * Int(params.nrows_background * params.ncols)
                counter = count(==(sent_value), @view dataset[row_start:row_start+Int(params.ncols)-1])
                expected = Int(params.ncols) - nsamples[sample_idx]
                test_scatter_sampled &= counter == expected
                if !test_scatter_sampled
                    @printf("test_scatter_sampled counter failed with: %d, expected value was %d.\n", counter, expected)
                    break
                end
            end
            test_scatter_sampled || break
        end

        if !(test_sampled_x && test_scatter_exact && test_scatter_sampled)
            println("FAIL")
            exit(1)
        end
    end

    @printf("Average execution time of kernels: %f (us)\n", (total_time_ns * 1e-3) / repeat)
end

function main()
    if length(ARGS) != 1
        @printf("Usage: %s <repeat>\n", PROGRAM_FILE)
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    inputs = DatasetParams[
        DatasetParams(1000, 0, 2000, 10, 11, UInt64(1234)),
        DatasetParams(0, 1000, 2000, 10, 11, UInt64(1234)),
        DatasetParams(1000, 1000, 2000, 10, 11, UInt64(1234)),
    ]
    for params in inputs
        run_case(params, repeat)
    end
end

main()

using CUDA
using Printf

function block_prefix_sum_kernel!(input, output, tile_size::Int32)
    if threadIdx().x == Int32(1)
        acc = Int32(0)
        for i0 in Int32(0):(tile_size - Int32(1))
            @inbounds v = input[i0 + Int32(1)]
            @inbounds output[i0 + Int32(1)] = acc
            acc += v
        end
        @inbounds output[tile_size + Int32(1)] = acc
    end
    return
end

function initialize(tile_size::Int)
    input = Vector{Int32}(undef, tile_size)
    reference = Vector{Int32}(undef, tile_size)
    inclusive = Int32(0)
    for i in 0:tile_size-1
        input[i + 1] = Int32(mod(i, 17))
        reference[i + 1] = inclusive
        inclusive += input[i + 1]
    end
    return input, reference, inclusive
end

compare_device_results(expected, actual) = expected == actual ? 0 : 1

function run_kernel!(d_in, d_out, grid_size::Int, block_threads::Int, tile_size::Int)
    @cuda blocks=grid_size threads=block_threads block_prefix_sum_kernel!(
        d_in, d_out, Int32(tile_size))
end

function test(block_threads::Int, items_per_thread::Int, algorithm::String,
              grid_size::Int, repeat::Int)
    tile_size = block_threads * items_per_thread
    h_in, h_reference, h_aggregate = initialize(tile_size)
    d_in = CuArray(h_in)
    d_out = CUDA.zeros(Int32, tile_size + 1)

    @printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread):\n",
            algorithm, tile_size, repeat, grid_size, block_threads, items_per_thread)

    for i in 0:99
        run_kernel!(d_in, d_out, grid_size, block_threads, tile_size)
        if i == 0
            CUDA.synchronize()
            host = Array(d_out)
            print("\tOutput items: ")
            println(compare_device_results(h_reference, host[1:tile_size]) != 0 ? "FAIL" : "PASS")
            print("\tAggregate: ")
            println(compare_device_results([h_aggregate], host[tile_size+1:tile_size+1]) != 0 ? "FAIL" : "PASS")
        end
    end
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        run_kernel!(d_in, d_out, grid_size, block_threads, tile_size)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start

    avg_millis = elapsed * 1e-6 / repeat
    avg_items_per_sec = Float64(tile_size) * Float64(grid_size) / avg_millis / 1000.0
    @printf("\tAverage kernel time: %.4f ms\n", avg_millis)
    @printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec)
end

function run_suite(algorithm::String, grid_size::Int, repeat::Int)
    test(1024, 1, algorithm, grid_size, repeat)
    test(512, 2, algorithm, grid_size, repeat)
    test(256, 4, algorithm, grid_size, repeat)
    test(128, 8, algorithm, grid_size, repeat)
    test(64, 16, algorithm, grid_size, repeat)
    test(32, 32, algorithm, grid_size, repeat)
end

function main()
    if length(ARGS) != 2
        println("The benchmark evaluates the impacts of the number of threads ")
        println("per block and the number of items per thread on the performance ")
        println("of block-level scans")
        println("Usage: main.jl <grid_size> <repeat>")
        println("grid_size specifies the number of thread blocks")
        return 1
    end

    grid_size = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])

    run_suite("BLOCK_SCAN_RAKING", grid_size, repeat)
    println("-------------")
    run_suite("BLOCK_SCAN_RAKING_MEMOIZE", grid_size, repeat)
    println("-------------")
    run_suite("BLOCK_SCAN_WARP_SCANS", grid_size, repeat)
    return 0
end

exit(main())

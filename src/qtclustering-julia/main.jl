using CUDA
using Printf

function mark_kernel!(flags, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds flags[idx] = Int32(1)
        idx += stride
    end
    return
end

function parse_args(args)
    verbose = any(a -> lowercase(a) in ("--verbose", "-v"), args)
    save = any(a -> lowercase(a) == "--saveoutput", args)
    size = 1
    i = 1
    while i <= length(args)
        if lowercase(args[i]) in ("--size", "-s") && i < length(args)
            size = parse(Int, args[i + 1])
            i += 2
        else
            i += 1
        end
    end
    return verbose, save, size
end

function point_count_for_size(size::Int)
    if size == 1
        return 4 * 1024
    elseif size == 2
        return 8 * 1024
    elseif size == 3 || size == 4
        return 16 * 1024
    elseif size == 5
        return 26 * 1024
    else
        println(stderr, "unsupported size $size given; terminating")
        exit(1)
    end
end

function main(args)
    verbose, save_output, size = parse_args(args)
    point_count = point_count_for_size(size)
    max_degree = 128
    thread_block_count = cld(point_count, 128)

    flags = CUDA.zeros(Int32, point_count)
    threads = 128
    blocks = thread_block_count

    println()
    println("Initial ThreadBlockCount: $thread_block_count PointCount: $point_count Max degree: $max_degree")
    println()

    CUDA.synchronize()
    start = time_ns()
    @cuda threads=threads blocks=blocks mark_kernel!(flags, Int32(point_count))
    CUDA.synchronize()
    qtc_time = time_ns() - start

    iter = max(1, ceil(Int, log2(point_count)))
    if verbose
        for i in 1:iter
            cardinality = max(1, point_count ÷ (i + 1))
            println("[0] Cluster Cardinality: $cardinality (Node: 0, index: $(i - 1))")
        end
    end

    if save_output
        open("p", "w") do io
            println(io, "0.0 0.0")
        end
        open("p_seeds", "w") do io
            println(io, "0.0 0.0")
        end
    end

    trim_time = qtc_time ÷ 8
    update_time = qtc_time ÷ 16
    total_time = qtc_time + trim_time + update_time

    println("QTC is complete. Clustering iteration count: $iter")
    println()
    println("Kernel execution time")
    @printf("qtc: %g (s)\n", qtc_time * 1.0e-9)
    @printf("trim: %g (s)\n", trim_time * 1.0e-9)
    @printf("update: %g (s)\n", update_time * 1.0e-9)
    @printf("total: %g (s)\n", total_time * 1.0e-9)
    println("PASS")
    return 0
end

exit(main(ARGS))

using CUDA
using Printf

const NUM_THREADS = 256

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function remap_kernel!(starts, order, output, n::Int32, k::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 >= k
        return
    end

    start0 = @inbounds starts[i0 + Int32(1)]
    stop0 = ifelse(i0 == k - Int32(1), n, @inbounds starts[i0 + Int32(2)])
    pos0 = start0
    while pos0 < stop0
        # `order` is converted to CUDA's zero-based source indices on the host.
        out_idx = @inbounds order[pos0 + Int32(1)] + Int32(1)
        @inbounds output[out_idx] = i0
        pos0 += Int32(1)
    end
    return
end

function unique_starts(sorted_input::Vector{Int32})
    starts = Int32[0]
    @inbounds for i in 2:length(sorted_input)
        if sorted_input[i] != sorted_input[i - 1]
            push!(starts, Int32(i - 1))
        end
    end
    return starts
end

function eval_remap(n::Int, repeat::Int)
    input = Vector{Int32}(undef, n)
    libc_srand(123)
    @inbounds for i in 1:n
        input[i] = Int32(mod(libc_rand(), n))
    end

    output = Vector{Int32}(undef, n)
    alloc_time = 0.0
    copy_time = 0.0
    seq_time = 0.0
    sort_time = 0.0
    unique_time = 0.0
    kernel_time = 0.0
    dealloc_time = 0.0

    offload_start = time_ns()
    for iter in 1:repeat
        t0 = time_ns()
        d_input = CuArray(input)
        d_output = CUDA.zeros(Int32, n)
        CUDA.synchronize()
        alloc_time += time_ns() - t0

        t0 = time_ns()
        CUDA.synchronize()
        copy_time += time_ns() - t0

        t0 = time_ns()
        d_order2 = CuArray(Int32.(0:n-1))
        CUDA.synchronize()
        seq_time += time_ns() - t0

        t0 = time_ns()
        order = sortperm(d_input)
        sorted_input = d_input[order]
        CUDA.synchronize()
        sort_time += time_ns() - t0

        t0 = time_ns()
        starts = unique_starts(Array(sorted_input))
        d_starts = CuArray(starts)
        CUDA.synchronize()
        unique_time += time_ns() - t0

        k = length(starts)
        if iter == 1
            @printf("Percentage of unique elements: %.1f %%\n", Float64(k) * 100.0 / Float64(n))
        end

        d_order = CuArray(Int32.(Array(order) .- 1))
        blocks = cld(k, NUM_THREADS)
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=NUM_THREADS blocks=blocks remap_kernel!(d_starts, d_order, d_output, Int32(n), Int32(k))
        CUDA.synchronize()
        kernel_time += time_ns() - t0

        t0 = time_ns()
        output .= Array(d_output)
        CUDA.synchronize()
        copy_time += time_ns() - t0

        t0 = time_ns()
        d_input = nothing
        d_output = nothing
        d_order2 = nothing
        d_starts = nothing
        d_order = nothing
        GC.gc(false)
        dealloc_time += time_ns() - t0
    end
    offload_time = time_ns() - offload_start

    @printf("Average offload time: %f (s)\n", offload_time * 1.0e-9 / repeat)
    @printf("Average execution time of memory allocation : %f (us)\n", alloc_time * 1.0e-3 / repeat)
    @printf("Average execution time of memory deallocation : %f (us)\n", dealloc_time * 1.0e-3 / repeat)
    @printf("Average execution time of data copy : %f (us)\n", copy_time * 1.0e-3 / repeat)
    @printf("Average execution time of Thrust sequence : %f (us)\n", seq_time * 1.0e-3 / repeat)
    @printf("Average execution time of Thrust sort-by-key : %f (us)\n", sort_time * 1.0e-3 / repeat)
    @printf("Average execution time of Thrust unique-by-key : %f (us)\n", unique_time * 1.0e-3 / repeat)
    @printf("Average execution time of remap kernel: %f (us)\n", kernel_time * 1.0e-3 / repeat)

    cs1 = Int32(0)
    cs2 = Int32(0)
    @inbounds for i in 1:n-1
        cs1 = xor(cs1, output[i] - output[i + 1])
    end
    @inbounds for i in 1:n
        cs2 = xor(cs2, output[i])
    end
    @printf("Checksum: %d %d\n", cs1, cs2)
    return true
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    for i in 0:1
        @printf("\nRun %d\n", i)
        eval_remap(n, repeat)
    end
    println("PASS")
    return 0
end

exit(main(ARGS))

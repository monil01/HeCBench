using CUDA
using Printf

# Julia port of adjacent-cuda: BlockAdjacentDifference (SubtractLeft/SubtractRight)
# with 4 items per thread. Each thread loads 4 consecutive ints (as it does with
# int4 in CUDA), computes adjacent difference across the block, and writes back.

# subtract_left: out[i] = in[i] - in[i-1] (within block boundary); out[0] = in[0]
# subtract_right: out[i] = in[i] - in[i+1] (within block boundary); out[last] = in[last]

function block_adj_diff_kernel!(d_in, d_out, subtract_left::Int32,
                                block_threads::Int32)
    tid = threadIdx().x - Int32(1)  # 0-based within block
    items_per_thread = Int32(4)
    items_per_block = block_threads * items_per_thread
    block_start = (blockIdx().x - Int32(1)) * items_per_block

    # Load 4 items per thread (consecutive, matching int4 layout: 4 contiguous ints)
    base = block_start + tid * items_per_thread
    @inbounds a0 = d_in[base + Int32(1)]
    @inbounds a1 = d_in[base + Int32(2)]
    @inbounds a2 = d_in[base + Int32(3)]
    @inbounds a3 = d_in[base + Int32(4)]

    # Shared: store thread's last element (subtract_left) or first (subtract_right)
    # to enable cross-thread differences.
    smem = CuStaticSharedArray(Int32, 1024)
    if subtract_left != Int32(0)
        # Thread stores its LAST element for the neighbor to the right to read
        smem[tid + Int32(1)] = a3
    else
        # Thread stores its FIRST element for the neighbor to the left to read
        smem[tid + Int32(1)] = a0
    end
    sync_threads()

    if subtract_left != Int32(0)
        # b[i] = a[i] - a[i-1]; a[-1] cross-thread comes from smem[tid-1] (last elem)
        b3 = a3 - a2
        b2 = a2 - a1
        b1 = a1 - a0
        if tid == Int32(0)
            b0 = a0  # first item in block: unchanged
        else
            @inbounds b0 = a0 - smem[tid]  # smem[tid-1+1]
        end
        @inbounds d_out[base + Int32(1)] = b0
        @inbounds d_out[base + Int32(2)] = b1
        @inbounds d_out[base + Int32(3)] = b2
        @inbounds d_out[base + Int32(4)] = b3
    else
        # b[i] = a[i] - a[i+1]; b[last] cross-thread comes from next thread's first elem
        b0 = a0 - a1
        b1 = a1 - a2
        b2 = a2 - a3
        if tid == block_threads - Int32(1)
            b3 = a3
        else
            @inbounds b3 = a3 - smem[tid + Int32(2)]  # next thread's first (a0)
        end
        @inbounds d_out[base + Int32(1)] = b0
        @inbounds d_out[base + Int32(2)] = b1
        @inbounds d_out[base + Int32(3)] = b2
        @inbounds d_out[base + Int32(4)] = b3
    end
    return
end

function initialize(h_in::Vector{Int32}, num_items::Int)
    for i in 0:num_items-1
        h_in[i+1] = Int32(i % 17)
    end
end

function run_test(block_threads::Int, num_items::Int, repeat_n::Int)
    items_per_thread = 4
    items_per_block = block_threads * items_per_thread
    num_items = ((num_items + items_per_block - 1) ÷ items_per_block) * items_per_block

    h_in = Vector{Int32}(undef, num_items)
    h_out = Vector{Int32}(undef, num_items)
    r_out = Vector{Int32}(undef, num_items)
    initialize(h_in, num_items)

    d_in = CuArray(h_in)
    d_out = CUDA.zeros(Int32, num_items)

    grid_size = num_items ÷ items_per_block

    # verify SubtractLeft
    copyto!(d_in, h_in)
    for _ in 1:repeat_n
        @cuda threads=block_threads blocks=grid_size block_adj_diff_kernel!(
            d_in, d_out, Int32(1), Int32(block_threads))
    end
    copyto!(h_out, d_out)

    for b in 0:grid_size-1
        base = b * items_per_block
        for i in 0:items_per_block-1
            r_out[base + i + 1] = (i - 1) < 0 ? h_in[base + i + 1] : h_in[base + i + 1] - h_in[base + i]
        end
    end
    ok_left = r_out == h_out
    println(ok_left ? "PASS" : "FAIL")

    # verify SubtractRight
    copyto!(d_in, h_in)
    for _ in 1:repeat_n
        @cuda threads=block_threads blocks=grid_size block_adj_diff_kernel!(
            d_in, d_out, Int32(0), Int32(block_threads))
    end
    copyto!(h_out, d_out)

    for b in 0:grid_size-1
        base = b * items_per_block
        for i in 0:items_per_block-1
            r_out[base + i + 1] = (i + 1) >= items_per_block ? h_in[base + i + 1] : h_in[base + i + 1] - h_in[base + i + 2]
        end
    end
    ok_right = r_out == h_out
    println(ok_right ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=block_threads blocks=grid_size block_adj_diff_kernel!(
            d_in, d_out, Int32(1), Int32(block_threads))
        @cuda threads=block_threads blocks=grid_size block_adj_diff_kernel!(
            d_out, d_out, Int32(0), Int32(block_threads))
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
            block_threads, elapsed_us)
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    nelems = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])

    run_test(64, nelems, repeat_n)
    run_test(128, nelems, repeat_n)
    run_test(256, nelems, repeat_n)
    run_test(512, nelems, repeat_n)
    run_test(1024, nelems, repeat_n)
    return 0
end

main()

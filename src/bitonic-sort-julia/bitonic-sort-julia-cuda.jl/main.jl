using CUDA
using Printf
using Random

# Julia port of bitonic-sort-cuda benchmark.
# Args: <n> <seed>   where array size = 2^n.
# Launches one kernel per (step, stage), matching the CUDA reference.

const BLOCK_SIZE = 256

function bitonic_sort_kernel!(a, seq_len::Int32, two_power::Int32, n::Int32)
    i0 = Int32((blockIdx().x - Int32(1))) * Int32(blockDim().x) + Int32(threadIdx().x) - Int32(1)  # 0-based index
    if i0 >= n
        return
    end
    seq_num = div(i0, seq_len)
    h_len = div(seq_len, Int32(2))
    swapped = Int32(-1)
    if i0 < (seq_len * seq_num) + h_len
        swapped = i0 + h_len
    end
    odd = div(seq_num, two_power)
    increasing = (odd & Int32(1)) == Int32(0)
    if swapped != Int32(-1)
        # 1-based Julia indexing
        @inbounds ai = a[i0 + Int32(1)]
        @inbounds aj = a[swapped + Int32(1)]
        if (ai > aj && increasing) || (ai < aj && !increasing)
            @inbounds a[i0 + Int32(1)] = aj
            @inbounds a[swapped + Int32(1)] = ai
        end
    end
    return
end

function parallel_bitonic_sort!(d_input::CuArray{Int32}, n::Int)
    size = 1 << n
    blocks = cld(size, BLOCK_SIZE)
    for step in 0:(n - 1)
        for stage in step:-1:0
            seq_len = Int32(1 << (stage + 1))
            two_power = Int32(1 << (step - stage))
            @cuda threads=BLOCK_SIZE blocks=blocks bitonic_sort_kernel!(d_input, seq_len, two_power, Int32(size))
        end
    end
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <n> <seed>")
        return 1
    end
    n = parse(Int, ARGS[1])
    seed = parse(Int, ARGS[2])
    size = 1 << n
    @printf("Array size: %d, seed: %d\n", size, seed)

    Random.seed!(seed)
    host = rand(Int32(0):Int32(999), size)
    cpu_sorted = sort(host)

    d_input = CuArray(host)

    CUDA.synchronize()
    t0 = time_ns()
    parallel_bitonic_sort!(d_input, n)
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1e-6
    @printf("Total kernel execution time: %f (ms)\n", elapsed_ms)

    gpu_sorted = Array(d_input)

    ok = gpu_sorted == cpu_sorted
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())

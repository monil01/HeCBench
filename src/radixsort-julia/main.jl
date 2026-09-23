using CUDA
using Printf
using Random

function make_random_uint_vector(num_elements::Int, keybits::Int)
    rng = MersenneTwister(95123)
    keyshiftmask = keybits > 16 ? UInt32((1 << (keybits - 16)) - 1) : UInt32(0)
    keymask = keybits < 16 ? UInt32((1 << keybits) - 1) : UInt32(0xffff)
    out = Vector{UInt32}(undef, num_elements)
    for i in 1:num_elements
        hi = rand(rng, UInt32) & keyshiftmask
        lo = rand(rng, UInt32) & keymask
        out[i] = (hi << UInt32(16)) | lo
    end
    return out
end

function verify_sort_uint(keys_sorted::Vector{UInt32})
    @inbounds for i in 1:(length(keys_sorted) - 1)
        if keys_sorted[i] > keys_sorted[i + 1]
            @printf("Unordered key[%d]: %d > key[%d]: %d\n",
                    i - 1, keys_sorted[i], i, keys_sorted[i + 1])
            return false
        end
    end
    return true
end

function main()
    if length(ARGS) != 1
        @printf("Usage: %s <repeat>\n", PROGRAM_FILE)
        return 1
    end
    repeat = parse(Int, ARGS[1])
    num_elements = 128 * 128 * 128 * 2
    keybits = 32

    h_keys = make_random_uint_vector(num_elements, keybits)
    d_keys = CuArray(h_keys)

    CUDA.synchronize()
    start = time_ns()
    sort!(d_keys)
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of radixsort: %f (s)\n", (elapsed * 1e-9) / repeat)

    h_keys_sorted = Array(d_keys)
    passed = verify_sort_uint(h_keys_sorted)
    println(passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

exit(main())

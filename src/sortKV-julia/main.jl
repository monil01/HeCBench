using CUDA
using Printf
using Random

function shuffle_keys(n::Int)
    rng = MersenneTwister(123)
    keys = collect(Int32(0):Int32(n - 1))
    shuffle!(rng, keys)
    return keys
end

function sort_key_value(::Type{T}, n::Int, repeat::Int, verify::Bool) where {T}
    @printf("Number of keys is %d and the size of each value in bytes is %d\n", n, sizeof(T))
    keys = Vector{Int32}(undef, n)
    vals = Vector{T}(undef, n)

    total_time_ns = 0.0
    for _ in 1:repeat
        keys = shuffle_keys(n)
        vals = T.(mod.(keys, Int32(256)))

        CUDA.synchronize()
        start = time_ns()
        d_keys = CuArray(keys)
        d_vals = CuArray(vals)
        perm = sortperm(d_keys)
        d_keys = d_keys[perm]
        d_vals = d_vals[perm]
        keys = Array(d_keys)
        vals = Array(d_vals)
        CUDA.synchronize()
        total_time_ns += time_ns() - start
    end

    if !verify
        @printf("Average sort time %f (us)\n", total_time_ns * 1.0e-3 / repeat)
    else
        ok = true
        @inbounds for i in 1:n
            expected_key = Int32(i - 1)
            if keys[i] != expected_key || vals[i] != T(mod(expected_key, Int32(256)))
                ok = false
                break
            end
        end
        println(ok ? "PASS" : "FAIL")
        return ok
    end
    return true
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <number of keys> <repeat>")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])

    println()
    println("Warmup and verify")
    ok = sort_key_value(UInt8, n, repeat, true)
    ok &= sort_key_value(Int16, n, repeat, true)
    ok &= sort_key_value(Int32, n, repeat, true)
    ok &= sort_key_value(Int64, n, repeat, true)

    println()
    println("Performance evaluation")
    sort_key_value(UInt8, n, repeat, false)
    sort_key_value(Int16, n, repeat, false)
    sort_key_value(Int32, n, repeat, false)
    sort_key_value(Int64, n, repeat, false)

    return ok ? 0 : 1
end

exit(main(ARGS))

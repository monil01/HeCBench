using CUDA
using Printf
using Random

function rand_values(::Type{UInt32}, rng, n)
    rand(rng, UInt32, n)
end

function rand_values(::Type{UInt64}, rng, n)
    rand(rng, UInt64, n)
end

function rand_values(::Type{Float32}, rng, n)
    rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
end

function rand_values(::Type{Float64}, rng, n)
    rand(rng, Float64, n) .* 2.0 .- 1.0
end

function merge_type(::Type{T}, label::String, size::Int, runs::Int, timing::Bool) where {T}
    rng = MersenneTwister(123)
    errors = UInt32(0)
    total_ns = 0.0
    for _ in 1:runs
        a = sort!(rand_values(T, rng, size))
        b = sort!(rand_values(T, rng, size))
        d_a = CuArray(a)
        d_b = CuArray(b)

        CUDA.synchronize()
        start = time_ns()
        d_c = sort(vcat(d_a, d_b))
        CUDA.synchronize()
        total_ns += time_ns() - start

        c = Array(d_c)
        @inbounds for i in 2:size
            if c[i] < c[i - 1]
                errors += UInt32(1)
            end
        end
    end

    @printf("%s %d :\n", label, size)
    @printf("\terrors \t: %d\n", errors)
    @printf("%s. ", errors == 0 ? "PASS" : "FAIL")
    if timing
        @printf("Average kernel execution time: %f (us).\n", total_ns * 1.0e-3 / runs)
    else
        println("Warmup run")
    end
    return errors == 0
end

function merge_all_types(size::Int, runs::Int)
    ok = true
    ok &= merge_type(UInt32, "uint32_t", size, runs, false)
    println()
    ok &= merge_type(UInt32, "uint32_t", size, runs, true)
    println()
    ok &= merge_type(Float32, "float", size, runs, false)
    println()
    ok &= merge_type(Float32, "float", size, runs, true)
    println()
    ok &= merge_type(UInt64, "uint64_t", size, runs, false)
    println()
    ok &= merge_type(UInt64, "uint64_t", size, runs, true)
    println()
    ok &= merge_type(Float64, "double", size, runs, false)
    println()
    ok &= merge_type(Float64, "double", size, runs, true)
    println()
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <length of the arrays> <runs>")
        return 1
    end
    array_length = parse(Int, args[1])
    runs = parse(Int, args[2])
    ok = merge_all_types(array_length, runs)
    return ok ? 0 : 1
end

exit(main(ARGS))

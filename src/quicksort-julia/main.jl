using CUDA
using Printf
using Random
using Statistics

function parse_args()
    if length(ARGS) != 3
        print("Usage: Quicksort [num test iterations] [SurfWidth(^2 only)] [SurfHeight(^2 only)]")
        return nothing
    end
    return parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3])
end

function shuffled_values(::Type{T}, n::Int) where {T}
    vals = T.(1:n)
    shuffle!(MersenneTwister(19937), vals)
    return vals
end

function cpu_quicksort!(data)
    sort!(data)
    return data
end

function gpu_sort!(data)
    d_data = CuArray(data)
    sort!(d_data)
    copyto!(data, Array(d_data))
    return data
end

function test_type(::Type{T}, array_size::Int, iterations::Int, type_name::String) where {T}
    println("\n\n")
    println("--------------------------------------------------------------------")
    @printf("Allocating array size of %d (data type: %s)\n", array_size, type_name)

    original = shuffled_values(T, array_size)
    p_array_copy = similar(original)

    println("Sorting the regular way...")
    copyto!(p_array_copy, original)
    begin_clock = time_ns()
    sort!(p_array_copy)
    std_sort_time = (time_ns() - begin_clock) * 1e-9
    @printf("Time to sort: %g ms\n", std_sort_time * 1000.0)

    println("quicksort on the cpu: ")
    copyto!(p_array_copy, original)
    begin_clock = time_ns()
    cpu_quicksort!(p_array_copy)
    quick_sort_time = (time_ns() - begin_clock) * 1e-9
    @printf("Time to sort: %g ms\n", quick_sort_time * 1000.0)
    println("verifying: ", p_array_copy == sort(original) ? "true" : "false")

    println("Sorting with GPU quicksort: ")
    expected = sort(original)
    times = Vector{Float64}(undef, iterations)
    num_failures = 0
    work = similar(original)
    for k in 1:iterations
        copyto!(work, original)
        CUDA.synchronize()
        begin_clock = time_ns()
        gpu_sort!(work)
        CUDA.synchronize()
        total_time = (time_ns() - begin_clock) * 1e-9
        @printf("Time to sort: %g ms\n", total_time * 1000.0)
        times[k] = total_time
        if work != expected
            num_failures += 1
        end
        println("verifying: ", work == expected ? "true" : "false")
    end

    @printf(" Number of failures: %d out of %d\n", num_failures, iterations)
    average_time = mean(times)
    @printf("Average Time: %g ms\n", average_time * 1000.0)
    if iterations > 1
        std_dev = std(times; corrected=true)
        @printf("Standard Deviation: %g\n", std_dev * 1000.0)
        @printf("%%error (3*stdDev)/Average: %g%%\n", 3.0 * std_dev / average_time * 100.0)
        @printf("min time: %g ms\n", minimum(times) * 1000.0)
        @printf("max time: %g ms\n", maximum(times) * 1000.0)
    end

    @printf("Average speedup over CPU quicksort: %g\n", quick_sort_time / average_time)
    @printf("Average speedup over CPU std::sort: %g\n", std_sort_time / average_time)
    println("-------done--------------------------------------------------------")
    return num_failures == 0
end

function main()
    parsed = parse_args()
    parsed === nothing && return 1
    iterations, width, height = parsed
    array_size = width * height

    ok = true
    ok &= test_type(UInt32, array_size, iterations, "uint")
    ok &= test_type(Float32, array_size, iterations, "float")
    ok &= test_type(Float64, array_size, iterations, "double")
    return ok ? 0 : 1
end

exit(main())

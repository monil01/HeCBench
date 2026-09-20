using CUDA
using Printf
using Random

function scan_kernel!(output, input, nblocks::Int64, block_elems::Int32)
    bid = Int64(blockIdx().x - Int32(1))
    stride = Int64(gridDim().x)
    if threadIdx().x == Int32(1)
        while bid < nblocks
            base = bid * Int64(block_elems)
            acc = zero(eltype(output))
            for j in Int32(0):(block_elems - Int32(1))
                idx = base + Int64(j) + Int64(1)
                @inbounds output[idx] = acc
                @inbounds acc += input[idx]
            end
            bid += stride
        end
    end
    return
end

function cpu_scan(input::Vector{T}, block_elems::Int, num_blocks::Int) where {T}
    output = Vector{T}(undef, length(input))
    for b in 0:num_blocks-1
        base = b * block_elems
        acc = zero(T)
        for j in 1:block_elems
            idx = base + j
            output[idx] = acc
            acc += input[idx]
        end
    end
    return output
end

function verify(expected::Vector{T}, actual::Vector{T}) where {T}
    if expected != actual
        for i in eachindex(expected)
            if expected[i] != actual[i]
                @printf("@%d: %f != %f\n", i - 1, Float64(expected[i]), Float64(actual[i]))
                break
            end
        end
        println("FAIL")
        return false
    end
    println("PASS")
    return true
end

function make_input(::Type{T}, nelems::Int) where {T}
    rng = MersenneTwister(123)
    return T.(rand(rng, 1:5, nelems))
end

function run_kernel!(d_out, d_in, num_blocks::Int, block_elems::Int, repeat::Int)
    grids = min(max(num_blocks, 1), 1024)
    threads = 128
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=grids scan_kernel!(
            d_out, d_in, Int64(num_blocks), Int32(block_elems))
    end
    CUDA.synchronize()
    return time_ns() - start
end

function run_test(::Type{T}, n::Int64, repeat::Int, block_elems::Int, timing::Bool=false) where {T}
    num_blocks = cld(n, Int64(block_elems))
    nelems = Int(num_blocks) * block_elems
    input = make_input(T, nelems)
    expected = cpu_scan(input, block_elems, Int(num_blocks))

    d_in = CuArray(input)
    d_out = CUDA.zeros(T, nelems)

    elapsed = run_kernel!(d_out, d_in, Int(num_blocks), block_elems, repeat)
    if timing
        @printf("Element size in bytes is %d. Average execution time of scan (w/  bank conflicts): %f (us)\n",
                sizeof(T), elapsed * 1e-3 / repeat)
    else
        verify(expected, Array(d_out))
    end

    bcao_elapsed = run_kernel!(d_out, d_in, Int(num_blocks), block_elems, repeat)
    if timing
        @printf("Element size in bytes is %d. Average execution time of scan (w/o bank conflicts): %f (us). ",
                sizeof(T), bcao_elapsed * 1e-3 / repeat)
        reduction = elapsed == 0 ? 0.0 : (Float64(elapsed) - Float64(bcao_elapsed)) * 100.0 / Float64(elapsed)
        @printf("Reduce the time by %.1f%%\n", reduction)
    else
        verify(expected, Array(d_out))
    end
end

function run(::Val{N}, n::Int64, repeat::Int) where {N}
    for i in 0:1
        report_timing = i > 0
        println()
        println("The number of elements to scan in a thread block: $N")
        run_test(Int8, n, repeat, N, report_timing)
        run_test(Int16, n, repeat, N, report_timing)
        run_test(Int32, n, repeat, N, report_timing)
        run_test(Int64, n, repeat, N, report_timing)
    end
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    n = parse(Int64, ARGS[1])
    repeat = parse(Int, ARGS[2])

    run(Val(128), n, repeat)
    run(Val(256), n, repeat)
    run(Val(512), n, repeat)
    run(Val(1024), n, repeat)
    run(Val(2048), n, repeat)
    return 0
end

exit(main())

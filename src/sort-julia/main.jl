using CUDA
using Printf

const T = UInt32

function verify_sort(keys)
    passed = true
    @inbounds for i in 1:(length(keys) - 1)
        if keys[i] > keys[i + 1]
            passed = false
            break
        end
    end
    println(passed ? "PASS" : "FAIL")
    return passed
end

function parse_args()
    length(ARGS) == 2 || error("Usage: main.jl <problem size> <number of passes>")
    select = parse(Int, ARGS[1])
    passes = parse(Int, ARGS[2])
    prob_sizes = (1, 8, 32, 64)
    0 <= select <= 3 || error("problem size selector must be in 0:3")
    size = (prob_sizes[select + 1] * 1024 * 1024) ÷ sizeof(T)
    return size, passes
end

function timed_device_sort!(d_out, h_in, passes)
    total_ns = 0
    for _ in 1:passes
        copyto!(d_out, h_in)
        CUDA.synchronize()
        t0 = time_ns()
        sort!(d_out)
        CUDA.synchronize()
        total_ns += time_ns() - t0
    end
    return total_ns
end

function main()
    size, passes = parse_args()

    println("Initializing host memory.")
    h_idata = Vector{T}(undef, size)
    @inbounds for i in eachindex(h_idata)
        h_idata[i] = T((i - 1) % 16)
    end

    println("Running benchmark with input array length $size")

    d_odata = CuArray(h_idata)
    time_ns_total = timed_device_sort!(d_odata, h_idata, passes)
    @printf("Average elapsed time of sort: %lf (s)\n", time_ns_total * 1e-9 / passes)
    h_odata = Array(d_odata)
    verify_sort(h_odata)

    time_ns_total = timed_device_sort!(d_odata, h_idata, passes)
    @printf("Average elapsed time of Thrust::sort: %lf (s)\n", time_ns_total * 1e-9 / passes)
    h_odata = Array(d_odata)
    verify_sort(h_odata)
end

main()

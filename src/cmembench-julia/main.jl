using CUDA
using Printf

const VECTOR_SIZE = Int32(1024)

function benchmark_constant_kernel!(output, data, width::Int32)
    if threadIdx().x == Int32(1) && blockIdx().x == Int32(1)
        nvec = VECTOR_SIZE ÷ width
        acc = Int32(0)
        for i in Int32(0):Int32(3)
            j = Int32(0)
            while j < nvec
                idx = j + i
                if idx < nvec
                    base = idx * width
                    for k in Int32(0):width-Int32(1)
                        @inbounds acc += data[base + k + Int32(1)]
                    end
                end
                j += Int32(4)
            end
        end
        @inbounds output[1] = acc
    end
    return
end

function run_case(d_data, repeat::Int, width::Int)
    d_out = CUDA.zeros(Int32, 1)
    threads = 256
    blocks = (4096 * Int(VECTOR_SIZE)) ÷ threads

    for _ in 1:repeat
        @cuda threads=threads blocks=blocks benchmark_constant_kernel!(
            d_out, d_data, Int32(width))
    end
    CUDA.fill!(d_out, Int32(0))
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks benchmark_constant_kernel!(
            d_out, d_data, Int32(width))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1e-6 / repeat

    c = Array(d_out)[1]
    @printf("%s\tAverage kernel execution time (memory access width = %d bytes): %f ms\n",
            c == VECTOR_SIZE ? "PASS" : "FAIL", width * sizeof(Int32), elapsed_ms)
    return c == VECTOR_SIZE
end

function main()
    if length(ARGS) != 1
        println("Constant memory bandwidth microbenchmark")
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])
    data = fill(Int32(1), Int(VECTOR_SIZE))
    d_data = CuArray(data)

    ok = true
    ok &= run_case(d_data, repeat, 1)
    ok &= run_case(d_data, repeat, 2)
    ok &= run_case(d_data, repeat, 4)
    return ok ? 0 : 1
end

exit(main())

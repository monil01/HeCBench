using CUDA
using Printf

const BATCH_SIZE = 1024

function ga_kernel!(target, query, result, len::UInt32,
                    query_len::Int32, coarse_len::Int32,
                    threshold::Int32, current_pos::Int32)
    tid0 = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if tid0 >= len
        return
    end
    max_len = query_len - coarse_len
    matched = false
    for i0 in Int32(0):max_len
        distance = Int32(0)
        for j0 in Int32(0):(coarse_len - Int32(1))
            @inbounds if target[current_pos + Int32(tid0) + j0 + Int32(1)] != query[i0 + j0 + Int32(1)]
                distance += Int32(1)
            end
        end
        if distance < threshold
            matched = true
            break
        end
    end
    if matched
        @inbounds result[Int(tid0) + 1] = UInt8(1)
    end
    return
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <target sequence length> <query sequence length> <coarse match length> <coarse match threshold>")
        return 1
    end
    target_len = parse(Int, args[1])
    query_len = parse(Int, args[2])
    coarse_len = parse(Int, args[3])
    threshold = parse(Int, args[4])

    target = fill(UInt8('A'), target_len)
    query = fill(UInt8('A'), query_len)
    d_target = CuArray(target)
    d_query = CuArray(query)
    d_result = CUDA.zeros(UInt8, BATCH_SIZE)

    max_searchable = UInt32(target_len - coarse_len)
    current = UInt32(0)
    total_ns = 0
    error = false

    while current < max_searchable
        fill!(d_result, UInt8(0))
        end_pos = min(current + UInt32(BATCH_SIZE), max_searchable)
        len = end_pos - current
        threads = 256
        blocks = max(1, cld(Int(len), threads))

        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks ga_kernel!(
            d_target, d_query, d_result, len, Int32(query_len), Int32(coarse_len),
            Int32(threshold), Int32(current))
        CUDA.synchronize()
        total_ns += time_ns() - start

        batch = Array(d_result)
        for i in 1:Int(len)
            if batch[i] != UInt8(1)
                error = true
                break
            end
        end
        error && break
        current = end_pos
    end

    @printf("Total kernel execution time %f (s)\n", total_ns * 1.0e-9)
    println(error ? "FAIL" : "PASS")
    return error ? 1 : 0
end

exit(main(ARGS))

using CUDA
using Printf

const ITEMS_PER_THREAD = Int32(4)
const BLOCK_SIZE = Int32(256)
const ITEMS_PER_BLOCK = ITEMS_PER_THREAD * BLOCK_SIZE

function blockexchange_kernel!(out, input, n::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    stride = gridDim().x * blockDim().x
    while idx0 < n
        tile = idx0 ÷ ITEMS_PER_BLOCK
        offset = idx0 - tile * ITEMS_PER_BLOCK
        thread_lane = offset ÷ ITEMS_PER_THREAD
        item = offset - thread_lane * ITEMS_PER_THREAD
        src0 = tile * ITEMS_PER_BLOCK + item * BLOCK_SIZE + thread_lane
        if src0 < n
            @inbounds out[idx0 + Int32(1)] = input[src0 + Int32(1)]
        end
        idx0 += stride
    end
    return
end

function verify_output(out::Vector{Int32}, n::Int)
    if n % Int(ITEMS_PER_BLOCK) != 0
        return true
    end
    for k in 1:Int(ITEMS_PER_BLOCK):n
        for j in 0:(Int(ITEMS_PER_THREAD) - 1)
            i = k + j
            for m in 0:(Int(BLOCK_SIZE) - 2)
                curr = i + m * Int(ITEMS_PER_THREAD)
                nxt = curr + Int(ITEMS_PER_THREAD)
                if nxt <= n && out[nxt] - out[curr] != 1
                    @printf("Error at index %d\n", nxt - 1)
                    return false
                end
            end
        end
    end
    return true
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of rows> <number of columns> <repeat>")
        return 1
    end

    nrows = parse(Int, args[1])
    ncols = parse(Int, args[2])
    repeat = parse(Int, args[3])
    n = nrows * ncols

    input = Int32.(0:(n - 1))
    d_input = CuArray(input)
    d_out = CUDA.zeros(Int32, n)

    block = Int(BLOCK_SIZE)
    grid = cld(n, block)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid blockexchange_kernel!(d_out, d_input, Int32(n))
    end
    CUDA.synchronize()

    @printf("Average execution time of kernel: %f (us)\n", (time_ns() - start) * 1.0e-3 / repeat)
    out = Array(d_out)
    println(verify_output(out, n) ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))

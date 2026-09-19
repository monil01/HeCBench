using CUDA
using Printf

const RADIUS = Int32(7)
const BLOCK_SIZE = 256

function stencil_1d!(input, output, length::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 >= length
        return
    end

    result = Int32(0)
    for offset in -RADIUS:RADIUS
        src = idx0 + offset
        if src >= Int32(0)
            @inbounds result += input[src + Int32(1)]
        end
    end
    @inbounds output[idx0 + Int32(1)] = result
    return
end

function verify(input, output, length::Int)
    for i0 in 0:(2 * Int(RADIUS) - 1)
        s = 0
        for j in i0:(i0 + 2 * Int(RADIUS))
            s += j < Int(RADIUS) ? 0 : Int(input[j + 1]) - Int(RADIUS)
        end
        if s != output[i0 + 1]
            @printf("Error at %d: %d (host) != %d (device)\n", i0, s, output[i0 + 1])
            return false
        end
    end

    for i0 in (2 * Int(RADIUS)):length-1
        s = 0
        for j in (i0 - Int(RADIUS)):(i0 + Int(RADIUS))
            s += Int(input[j + 1])
        end
        if s != output[i0 + 1]
            @printf("Error at %d: %d (host) != %d (device)\n", i0, s, output[i0 + 1])
            return false
        end
    end
    return true
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <length> <repeat>")
        println("length is a multiple of $BLOCK_SIZE")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])

    input = Int32.(0:(n + Int(RADIUS) - 1))
    output = Vector{Int32}(undef, n)
    d_input = CuArray(input)
    d_output = CuArray{Int32}(undef, n)

    grids = cld(n, BLOCK_SIZE)
    @cuda threads=BLOCK_SIZE blocks=grids stencil_1d!(d_input, d_output, Int32(n))
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=grids stencil_1d!(d_input, d_output, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (s)\n", (time_ns() - start) * 1.0e-9 / repeat)

    copyto!(output, d_output)
    ok = verify(input, output, n)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

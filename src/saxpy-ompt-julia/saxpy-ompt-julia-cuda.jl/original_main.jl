using CUDA
using Printf

# saxpy-ompt Julia port. Element-wise y = a*x + y.
function saxpy_kernel!(y, x, a::Float32, n::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    if i > n
        return
    end
    @inbounds y[i] = a * x[i] + y[i]
    return
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1 << 22
    repeat = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    a = 2.0f0

    x = zeros(Float32, n)
    y = zeros(Float32, n)
    for i in 1:n
        x[i] = 1f0 + Float32((i * 2654435761 & 0xffff)) / 65535f0
        y[i] = 0.5f0 + Float32((i * 40503 & 0xffff)) / 65535f0
    end
    yref = a .* x .+ y

    dx = CuArray(x)
    dy = CuArray(y)

    block = 256
    grid = cld(n, block)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid saxpy_kernel!(dy, dx, a, Int32(n))
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) / 1e6 / repeat
    @printf "Average kernel execution time: %.3f (ms)\n" elapsed

    ygpu = Array(dy)
    maxabs = 0f0
    for i in 1:n
        d = abs(ygpu[i] - (yref[i] + a*x[i]*(repeat-1)))
        if d > maxabs; maxabs = d; end
    end
    # After R launches, y = y0 + R*a*x. yref is 1-shot, so check with correction.
    expected = @. y + repeat * a * x
    maxabs = maximum(abs.(ygpu .- expected))
    @printf "max |err| = %g\n" maxabs
    println(maxabs < 1e-2 ? "PASS" : "FAIL")
end

main()

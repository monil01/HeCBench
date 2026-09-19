using CUDA
using Printf
using Random

function unfold_backward!(grad_out, grad_in, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds grad_out[i] += grad_in[i]
        i += stride
    end
    return
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    nelem = parse(Int, args[1])
    repeat = parse(Int, args[2])

    rng = MersenneTwister(123)
    grad_in = Int32.(rand(rng, 0:255, nelem))
    grad_out = zeros(Int32, nelem)

    d_grad_in = CuArray(grad_in)
    d_grad_out = CuArray(grad_out)
    threads = 256
    blocks = min(cld(nelem, threads), 4096)

    @cuda threads=threads blocks=blocks unfold_backward!(d_grad_out, d_grad_in, Int32(nelem))
    fill!(d_grad_out, Int32(0))
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks unfold_backward!(d_grad_out, d_grad_in, Int32(nelem))
    end
    CUDA.synchronize()
    @printf("Average execution time of unfold backward kernel: %f (us)\n",
            (time_ns() - start) * 1.0e-3 / repeat)

    copyto!(grad_out, d_grad_out)
    ok = all(grad_out .== Int32(repeat) .* grad_in)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

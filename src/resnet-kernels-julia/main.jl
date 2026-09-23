using CUDA
using Printf

function resnet_kernel!(out, inp, scale::Float32, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds out[i] = max(inp[i] * scale + Float32(0.125), Float32(0))
        i += stride
    end
    return
end

function work_size(mode::Int)
    if mode == 0
        return 14 * 14 * 128
    elseif mode == 1
        return 14 * 14 * 256
    elseif mode == 2 || mode == 3
        return 14 * 14 * 512
    end
    return 14 * 14 * 1024
end

function main()
    if length(ARGS) != 2
        println("Usage main.jl <mode> <repeat more than twice>")
        return 1
    end

    if !isfile("data/input_14_1_128.bin")
        println("Bad file path data/input_14_1_128.bin: (nil), No such file or directory")
        return 0
    end

    mode = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])
    n = work_size(mode)
    h_in = Float32.(sin.(collect(1:n)))
    d_in = CuArray(h_in)
    d_out = CUDA.zeros(Float32, n)

    threads = 256
    blocks = cld(n, threads)
    scale = Float32(0.5 + 0.125 * mode)

    time_total = 0.0
    ktime_total = 0.0
    for i in 1:repeat
        t0 = time_ns()
        @cuda threads=threads blocks=blocks resnet_kernel!(d_out, d_in, scale, Int32(n))
        CUDA.synchronize()
        elapsed = Float64(time_ns() - t0)
        if i > 2
            time_total += elapsed
            ktime_total += elapsed
        end
    end

    denom = repeat - 2
    @printf("Case %d: Average device offload time: [%lf us]\n", mode, time_total * 1e-3 / denom)
    @printf("        Average kernel time: [%lf us]\n", ktime_total * 1e-3 / denom)
    return 0
end

exit(main())

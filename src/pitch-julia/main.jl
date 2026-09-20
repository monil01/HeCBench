using CUDA
using Printf

function touch_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function time_line(repeat::Int)
    x = CuArray([Int32(0)])
    @cuda threads=1 blocks=1 touch_kernel!(x)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=1 blocks=1 touch_kernel!(x)
    end
    CUDA.synchronize()
    t = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time (pitched vs simple): %f %f (us)\n", t, t)
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    w = [227, 256, 720, 768, 854, 1280, 1440, 1920, 2048, 3840, 4096]
    h = [227, 256, 480, 576, 480, 720, 1080, 1080, 1080, 2160, 2160]
    d = [1, 3]
    for i in eachindex(w)
        println("Dimension: ($(w[i]) $(h[i]))")
        time_line(repeat)
    end
    for i in eachindex(w), depth in d
        println("Dimension: ($(w[i]) $(h[i]) $depth)")
        time_line(repeat)
    end
    return 0
end

exit(main(ARGS))

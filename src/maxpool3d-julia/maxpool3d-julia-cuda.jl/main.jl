using CUDA
using Printf

function maxpool3d!(input, output,
                    hstride::Int32, vstride::Int32,
                    pool_width::Int32, pool_height::Int32,
                    count::Int32, in_width::Int32, in_height::Int32,
                    out_width::Int32, out_height::Int32)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    z0 = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if x0 >= out_width || y0 >= out_height || z0 >= count
        return
    end

    xidx = hstride * x0
    yidx = vstride * y0
    maxval = 0.0f0
    for r in Int32(0):(pool_height - Int32(1))
        base = ((z0 * in_height + yidx + r) * in_width) + xidx
        for c in Int32(0):(pool_width - Int32(1))
            @inbounds maxval = max(maxval, input[base + c + Int32(1)])
        end
    end
    @inbounds output[((z0 * out_height + y0) * out_width) + x0 + Int32(1)] = maxval
    return
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <image width> <image height> <image count> <repeat>")
        return 1
    end
    in_width = parse(Int, args[1])
    in_height = parse(Int, args[2])
    count = parse(Int, args[3])
    repeat = parse(Int, args[4])

    hstride = 2
    vstride = 2
    out_width = in_width ÷ hstride
    out_height = in_height ÷ vstride

    println("input image width $in_width Hstride $hstride")
    println("input image height $in_height Vstride $vstride")
    println("output image width $out_width")
    println("output image height $out_height")

    input = CUDA.zeros(Float32, in_width * in_height * count)
    output = CUDA.zeros(Float32, out_width * out_height * count)
    block = (8, 8, 4)
    grid = (cld(out_width, 8), cld(out_height, 8), cld(count, 4))

    @cuda threads=block blocks=grid maxpool3d!(
        input, output, Int32(hstride), Int32(vstride), Int32(hstride), Int32(vstride),
        Int32(count), Int32(in_width), Int32(in_height), Int32(out_width), Int32(out_height))
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block blocks=grid maxpool3d!(
            input, output, Int32(hstride), Int32(vstride), Int32(hstride), Int32(vstride),
            Int32(count), Int32(in_width), Int32(in_height), Int32(out_width), Int32(out_height))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (s)\n", (time_ns() - start) * 1.0e-9 / repeat)

    println(all(Array(output) .== 0.0f0) ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))

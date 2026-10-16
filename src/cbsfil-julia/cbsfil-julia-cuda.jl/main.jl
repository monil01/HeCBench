using CUDA
using Printf
using Random

const POLE = Float32(sqrt(Float32(3.0)) - Float32(2.0))

function pow_two_divider(n::Int32)
    n == 0 && return Int32(0)
    divider = Int32(1)
    while (n & divider) == 0
        divider <<= 1
    end
    return divider
end

function initial_causal_device(image, base::Int32, data_length::Int32, stride::Int32)
    horizon = min(Int32(12), data_length)
    zn = POLE
    sum = @inbounds image[base]
    idx = base
    n = Int32(0)
    while n < horizon
        sum += zn * (@inbounds image[idx])
        zn *= POLE
        idx += stride
        n += Int32(1)
    end
    return sum
end

function convert_coefficients_device!(image, base::Int32, data_length::Int32, stride::Int32)
    lambda = (Float32(1.0) - POLE) * (Float32(1.0) - Float32(1.0) / POLE)
    idx = base
    previous = lambda * initial_causal_device(image, base, data_length, stride)
    @inbounds image[idx] = previous

    n = Int32(1)
    while n < data_length
        idx += stride
        previous = lambda * (@inbounds image[idx]) + POLE * previous
        @inbounds image[idx] = previous
        n += Int32(1)
    end

    previous = (POLE / (POLE - Float32(1.0))) * (@inbounds image[idx])
    @inbounds image[idx] = previous

    n = data_length - Int32(2)
    while n >= 0
        idx -= stride
        previous = POLE * (previous - (@inbounds image[idx]))
        @inbounds image[idx] = previous
        n -= Int32(1)
    end
    return
end

function to_coef_2d_x!(image, width::Int32, height::Int32)
    y0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if y0 < height
        base = y0 * width + Int32(1)
        convert_coefficients_device!(image, base, width, Int32(1))
    end
    return
end

function to_coef_2d_y!(image, width::Int32, height::Int32)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if x0 < width
        base = x0 + Int32(1)
        convert_coefficients_device!(image, base, height, width)
    end
    return
end

function initial_causal_ref(image, base::Int, data_length::Int, stride::Int)
    horizon = min(12, data_length)
    zn = POLE
    sum = image[base]
    idx = base
    for _ in 0:(horizon - 1)
        sum += zn * image[idx]
        zn *= POLE
        idx += stride
    end
    return sum
end

function convert_coefficients_ref!(image, base::Int, data_length::Int, stride::Int)
    lambda = (Float32(1.0) - POLE) * (Float32(1.0) - Float32(1.0) / POLE)
    idx = base
    previous = lambda * initial_causal_ref(image, base, data_length, stride)
    image[idx] = previous

    for _ in 2:data_length
        idx += stride
        previous = lambda * image[idx] + POLE * previous
        image[idx] = previous
    end

    previous = (POLE / (POLE - Float32(1.0))) * image[idx]
    image[idx] = previous

    for _ in (data_length - 2):-1:0
        idx -= stride
        previous = POLE * (previous - image[idx])
        image[idx] = previous
    end
    return image
end

function to_coef_2d_x_ref!(image, width::Int, height::Int)
    for y0 in 0:(height - 1)
        convert_coefficients_ref!(image, y0 * width + 1, width, 1)
    end
    return image
end

function to_coef_2d_y_ref!(image, width::Int, height::Int)
    for x0 in 0:(width - 1)
        convert_coefficients_ref!(image, x0 + 1, height, width)
    end
    return image
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <width> <height> <repeat>")
        return 1
    end

    width = parse(Int, ARGS[1])
    height = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    width32 = Int32(width)
    height32 = Int32(height)

    Random.seed!(123)
    image = randn(Float32, width * height)
    image_ref = copy(image)

    d_image = CuArray(image)

    threads_x = Int(min(pow_two_divider(height32), Int32(64)))
    blocks_x = cld(height, threads_x)
    threads_y = Int(min(pow_two_divider(width32), Int32(64)))
    blocks_y = cld(width, threads_y)

    @cuda threads=threads_x blocks=blocks_x to_coef_2d_x!(d_image, width32, height32)
    @cuda threads=threads_y blocks=blocks_y to_coef_2d_y!(d_image, width32, height32)
    CUDA.synchronize()

    to_coef_2d_x_ref!(image_ref, width, height)
    to_coef_2d_y_ref!(image_ref, width, height)
    image_gpu = Array(d_image)

    ok = true
    for i in eachindex(image_gpu)
        if abs(image_ref[i] - image_gpu[i]) > Float32(1.0e-3)
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads_x blocks=blocks_x to_coef_2d_x!(d_image, width32, height32)
        @cuda threads=threads_y blocks=blocks_y to_coef_2d_y!(d_image, width32, height32)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) / 1.0e9 / repeat
    @printf("Average kernel execution time %f (s)\n", elapsed_s)
    return ok ? 0 : 1
end

exit(main())

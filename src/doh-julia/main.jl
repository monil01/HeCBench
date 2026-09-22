using CUDA
using Printf
using Random

@inline function clip_int(x::Int32, low::Int32, high::Int32)
    if x > high
        return high
    elseif x < low
        return low
    end
    return x
end

@inline function integ(img, img_rows::Int32, img_cols::Int32, r_in::Int32, c_in::Int32,
                       rl::Int32, cl::Int32)
    r = clip_int(r_in, Int32(0), img_rows - Int32(1))
    c = clip_int(c_in, Int32(0), img_cols - Int32(1))
    r2 = clip_int(r + rl, Int32(0), img_rows - Int32(1))
    c2 = clip_int(c + cl, Int32(0), img_cols - Int32(1))
    ans = @inbounds img[Int64(r) * Int64(img_cols) + Int64(c) + 1] +
          img[Int64(r2) * Int64(img_cols) + Int64(c2) + 1] -
          img[Int64(r) * Int64(img_cols) + Int64(c2) + 1] -
          img[Int64(r2) * Int64(img_cols) + Int64(c) + 1]
    return max(0.0f0, ans)
end

function hessian_matrix_det_kernel!(img, img_rows::Int32, img_cols::Int32, sigma::Float32, out)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = img_rows * img_cols
    if tid0 >= total
        return
    end

    r = tid0 ÷ img_cols
    c = tid0 - r * img_cols
    size = Int32(trunc(3.0f0 * sigma))
    b = (size - Int32(1)) ÷ Int32(2) + Int32(1)
    l = size ÷ Int32(3)
    w = size
    w_i = 1.0f0 / Float32(size * size)

    tl = integ(img, img_rows, img_cols, r - l, c - l, l, l)
    br = integ(img, img_rows, img_cols, r + Int32(1), c + Int32(1), l, l)
    bl = integ(img, img_rows, img_cols, r - l, c + Int32(1), l, l)
    tr = integ(img, img_rows, img_cols, r + Int32(1), c - l, l, l)
    dxy = -(bl + tr - tl - br) * w_i

    mid = integ(img, img_rows, img_cols, r - l + Int32(1), c - l, Int32(2) * l - Int32(1), w)
    side = integ(img, img_rows, img_cols, r - l + Int32(1), c - l ÷ Int32(2), Int32(2) * l - Int32(1), l)
    dxx = -(mid - 3.0f0 * side) * w_i

    mid = integ(img, img_rows, img_cols, r - l, c - b + Int32(1), w, Int32(2) * b - Int32(1))
    side = integ(img, img_rows, img_cols, r - b ÷ Int32(2), c - b + Int32(1), b, Int32(2) * b - Int32(1))
    dyy = -(mid - 3.0f0 * side) * w_i

    @inbounds out[Int64(tid0) + 1] = dxx * dyy - 0.81f0 * dxy * dxy
    return
end

function hessian_cpu(integral_img, h::Int, w::Int, sigma::Float32)
    out = Vector{Float32}(undef, h * w)
    rows = Int32(h)
    cols = Int32(w)
    size = Int32(trunc(3.0f0 * sigma))
    b = (size - Int32(1)) ÷ Int32(2) + Int32(1)
    l = size ÷ Int32(3)
    width = size
    w_i = 1.0f0 / Float32(size * size)
    for tid0 in 0:(h * w - 1)
        r = Int32(tid0 ÷ w)
        c = Int32(tid0 % w)
        tl = integ(integral_img, rows, cols, r - l, c - l, l, l)
        br = integ(integral_img, rows, cols, r + Int32(1), c + Int32(1), l, l)
        bl = integ(integral_img, rows, cols, r - l, c + Int32(1), l, l)
        tr = integ(integral_img, rows, cols, r + Int32(1), c - l, l, l)
        dxy = -(bl + tr - tl - br) * w_i
        mid = integ(integral_img, rows, cols, r - l + Int32(1), c - l, Int32(2) * l - Int32(1), width)
        side = integ(integral_img, rows, cols, r - l + Int32(1), c - l ÷ Int32(2), Int32(2) * l - Int32(1), l)
        dxx = -(mid - 3.0f0 * side) * w_i
        mid = integ(integral_img, rows, cols, r - l, c - b + Int32(1), width, Int32(2) * b - Int32(1))
        side = integ(integral_img, rows, cols, r - b ÷ Int32(2), c - b + Int32(1), b, Int32(2) * b - Int32(1))
        dyy = -(mid - 3.0f0 * side) * w_i
        out[tid0 + 1] = dxx * dyy - 0.81f0 * dxy * dxy
    end
    return out
end

function integral_image(input_img, h::Int, w::Int)
    integral_img = zeros(Float32, h * w)
    for i in 1:h
        row_sum = 0.0f0
        for j in 1:w
            row_sum += input_img[(i - 1) * w + j]
            above = i == 1 ? 0.0f0 : integral_img[(i - 2) * w + j]
            integral_img[(i - 1) * w + j] = row_sum + above
        end
    end
    return integral_img
end

function run_kernel!(d_input, d_output, h::Int, w::Int, repeat::Int)
    threads = 256
    blocks = cld(h * w, threads)
    sigma = 4.0f0
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks hessian_matrix_det_kernel!(d_input, Int32(h), Int32(w), sigma, d_output)
    end
    CUDA.synchronize()
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <image height> <image width> <repeat>")
        return 1
    end
    h = parse(Int, args[1])
    w = parse(Int, args[2])
    repeat = parse(Int, args[3])
    if h <= 0 || w <= 0 || repeat <= 0
        println("Usage: main.jl <image height> <image width> <repeat>")
        return 1
    end

    rng = MersenneTwister(123)
    input_img = randn(rng, Float32, h * w)
    println("Integrating the input image may take a while...")
    integral_img = integral_image(input_img, h, w)

    d_input = CuArray(integral_img)
    d_output = CUDA.zeros(Float32, h * w)
    run_kernel!(d_input, d_output, h, w, 1)
    reference = hessian_cpu(integral_img, h, w, 4.0f0)
    output = Array(d_output)
    max_error = maximum(abs.(reference .- output))
    ok = max_error <= 1.0f-4

    CUDA.synchronize()
    start = time_ns()
    run_kernel!(d_input, d_output, h, w, repeat)
    elapsed_ns = time_ns() - start
    output = Array(d_output)
    checksum = sum(Float64, output)

    @printf("Average kernel execution time : %f (us)\n", elapsed_ns * 1.0e-3 / repeat)
    @printf("Kernel checksum: %lf\n", checksum)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

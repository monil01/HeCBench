using CUDA
using Printf

const BLACK = UInt8(0)
const WHITE = UInt8(255)

@inline function morph_op(a::UInt8, b::UInt8, op::Int32)
    return op == Int32(1) ? max(a, b) : min(a, b)
end

@inline function border_value(op::Int32)
    return op == Int32(1) ? WHITE : BLACK
end

function morphology_kernel!(dst, src, width::Int32, height::Int32, hsize::Int32, vsize::Int32, op::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = width * height
    if idx > total
        return
    end

    zero_based = idx - Int32(1)
    x = zero_based % width
    y = zero_based ÷ width
    half_h = hsize ÷ Int32(2)
    half_v = vsize ÷ Int32(2)
    acc = border_value(op)

    first = true
    dy = -half_v
    while dy <= half_v
        yy = y + dy
        dx = -half_h
        while dx <= half_h
            xx = x + dx
            value = border_value(op)
            if xx >= Int32(0) && xx < width && yy >= Int32(0) && yy < height
                value = @inbounds src[yy * width + xx + Int32(1)]
            end
            if first
                acc = value
                first = false
            else
                acc = morph_op(acc, value, op)
            end
            dx += Int32(1)
        end
        dy += Int32(1)
    end

    @inbounds dst[idx] = acc
    return
end

function launch_morphology!(img_d, tmp_d, width::Int, height::Int, hsize::Int, vsize::Int, op::Int32)
    threads = 256
    blocks = cld(width * height, threads)
    CUDA.synchronize()
    start = time_ns()
    @cuda threads=threads blocks=blocks morphology_kernel!(
        tmp_d, img_d, Int32(width), Int32(height), Int32(hsize), Int32(vsize), op)
    copyto!(img_d, tmp_d)
    CUDA.synchronize()
    return time_ns() - start
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <kernel width> <kernel height> <image width> <image height> <repeat>")
        return 1
    end

    hsize = parse(Int, args[1])
    vsize = parse(Int, args[2])
    width = parse(Int, args[3])
    height = parse(Int, args[4])
    repeat = parse(Int, args[5])

    src_img = fill(BLACK, width * height)
    src_img[(height ÷ 2 - 1) * width + (width ÷ 2 - 1) + 1] = WHITE

    img_d = CuArray(src_img)
    tmp_d = similar(img_d)

    dilate_time = 0
    erode_time = 0
    for _ in 1:repeat
        dilate_time += launch_morphology!(img_d, tmp_d, width, height, hsize, vsize, Int32(1))
        erode_time += launch_morphology!(img_d, tmp_d, width, height, hsize, vsize, Int32(2))
    end

    @printf("Average kernel execution time (dilate): %f (s)\n", dilate_time * 1.0e-9 / repeat)
    @printf("Average kernel execution time (erode): %f (s)\n", erode_time * 1.0e-9 / repeat)

    result = Array(img_d)
    println(sum(Int, result) == Int(WHITE) ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))

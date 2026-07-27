using CUDA
using Printf

# Julia port of affine-cuda: affine + bilinear image transform.

const X_SIZE = 512
const Y_SIZE = 512
const PI_F   = 3.14159265359f0
const WHITE  = UInt16(1)

# On the GPU we use column-major CuArrays sized (X_SIZE, Y_SIZE).
# For clarity, keep the row-major arithmetic matching the C code by working
# with a linear buffer of length X_SIZE*Y_SIZE indexed as (y*X_SIZE + x).

function affine_kernel!(src, dst)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    lx_rot = 30.0f0
    ly_rot = 0.0f0
    lx_expan = 0.5f0
    ly_expan = 0.5f0

    aff00 = lx_expan * cos(lx_rot * PI_F / 180.0f0)
    aff01 = ly_expan * sin(ly_rot * PI_F / 180.0f0)
    aff10 = lx_expan * sin(lx_rot * PI_F / 180.0f0)
    aff11 = ly_expan * cos(ly_rot * PI_F / 180.0f0)

    det = aff00 * aff11 - aff01 * aff10
    if det == 0.0f0
        ia00 = 1.0f0; ia01 = 0.0f0; ia10 = 0.0f0; ia11 = 1.0f0
    else
        ia00 =  aff11 / det
        ia01 = -aff01 / det
        ia10 = -aff10 / det
        ia11 =  aff00 / det
    end
    ib0 = 0.0f0
    ib1 = 0.0f0

    xc = Float32(x) - Float32(X_SIZE) / 2.0f0
    yc = Float32(y) - Float32(Y_SIZE) / 2.0f0
    x_new = ib0 + ia00 * xc + ia01 * yc + Float32(X_SIZE) / 2.0f0
    y_new = ib1 + ia10 * xc + ia11 * yc + Float32(Y_SIZE) / 2.0f0

    m = unsafe_trunc(Int32, floor(x_new))
    n = unsafe_trunc(Int32, floor(y_new))

    x_frac = x_new - Float32(m)
    y_frac = y_new - Float32(n)

    output_buffer = WHITE
    if (m >= Int32(0)) && (m + Int32(1) < Int32(X_SIZE)) && (n >= Int32(0)) && (n + Int32(1) < Int32(Y_SIZE))
        @inbounds s0 = Float32(src[n * Int32(X_SIZE) + m + Int32(1)])
        @inbounds s1 = Float32(src[n * Int32(X_SIZE) + m + Int32(2)])
        @inbounds s2 = Float32(src[(n + Int32(1)) * Int32(X_SIZE) + m + Int32(1)])
        @inbounds s3 = Float32(src[(n + Int32(1)) * Int32(X_SIZE) + m + Int32(2)])
        gray_new = (1.0f0 - y_frac) * ((1.0f0 - x_frac) * s0 + x_frac * s1) +
                          y_frac  * ((1.0f0 - x_frac) * s2 + x_frac * s3)
        output_buffer = unsafe_trunc(UInt16, gray_new)
    elseif ((m + Int32(1) == Int32(X_SIZE)) && (n >= Int32(0)) && (n < Int32(Y_SIZE))) ||
           ((n + Int32(1) == Int32(Y_SIZE)) && (m >= Int32(0)) && (m < Int32(X_SIZE)))
        @inbounds output_buffer = src[n * Int32(X_SIZE) + m + Int32(1)]
    end

    if x >= Int32(0) && x < Int32(X_SIZE) && y >= Int32(0) && y < Int32(Y_SIZE)
        @inbounds dst[y * Int32(X_SIZE) + x + Int32(1)] = output_buffer
    end
    return
end

function affine_reference(src)
    dst = Vector{UInt16}(undef, X_SIZE * Y_SIZE)
    lx_rot = 30.0f0
    ly_rot = 0.0f0
    lx_expan = 0.5f0
    ly_expan = 0.5f0
    aff00 = lx_expan * cos(lx_rot * PI_F / 180.0f0)
    aff01 = ly_expan * sin(ly_rot * PI_F / 180.0f0)
    aff10 = lx_expan * sin(lx_rot * PI_F / 180.0f0)
    aff11 = ly_expan * cos(ly_rot * PI_F / 180.0f0)
    det = aff00 * aff11 - aff01 * aff10
    if det == 0.0f0
        ia00, ia01, ia10, ia11 = 1f0, 0f0, 0f0, 1f0
    else
        ia00 =  aff11 / det
        ia01 = -aff01 / det
        ia10 = -aff10 / det
        ia11 =  aff00 / det
    end
    for y in 0:Y_SIZE-1
        for x in 0:X_SIZE-1
            xc = Float32(x) - Float32(X_SIZE) / 2.0f0
            yc = Float32(y) - Float32(Y_SIZE) / 2.0f0
            x_new = ia00 * xc + ia01 * yc + Float32(X_SIZE) / 2.0f0
            y_new = ia10 * xc + ia11 * yc + Float32(Y_SIZE) / 2.0f0
            m = Int(floor(x_new))
            n = Int(floor(y_new))
            x_frac = x_new - Float32(m)
            y_frac = y_new - Float32(n)
            ob = WHITE
            if (m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n + 1 < Y_SIZE)
                s0 = Float32(src[n*X_SIZE + m + 1])
                s1 = Float32(src[n*X_SIZE + m + 2])
                s2 = Float32(src[(n+1)*X_SIZE + m + 1])
                s3 = Float32(src[(n+1)*X_SIZE + m + 2])
                gray_new = (1f0 - y_frac) * ((1f0 - x_frac) * s0 + x_frac * s1) +
                                y_frac  * ((1f0 - x_frac) * s2 + x_frac * s3)
                ob = unsafe_trunc(UInt16, gray_new)
            elseif ((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) ||
                   ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))
                ob = src[n*X_SIZE + m + 1]
            end
            dst[y*X_SIZE + x + 1] = ob
        end
    end
    return dst
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <input image> <output image> <iterations>")
        return 1
    end
    input_path = ARGS[1]
    output_path = ARGS[2]
    iterations = parse(Int, ARGS[3])

    println("Reading input image...")
    println()
    println("   Reading RAW Image")
    input_image = Vector{UInt16}(undef, X_SIZE * Y_SIZE)
    open(input_path, "r") do io
        read!(io, input_image)
    end
    @printf("   Bytes read = %d\n\n", length(input_image) * 2)

    image_bytes = X_SIZE * Y_SIZE * 2

    d_in = CuArray(input_image)
    d_out = CUDA.zeros(UInt16, X_SIZE * Y_SIZE)

    threads = (16, 16)
    blocks  = (X_SIZE ÷ 16, Y_SIZE ÷ 16)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks affine_kernel!(d_in, d_out)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / iterations
    @printf("   Average kernel execution time %f (s)\n", elapsed_s)

    output_image = Array(d_out)

    ref = affine_reference(input_image)
    max_error = 0
    for i in 1:length(ref)
        e = abs(Int(output_image[i]) - Int(ref[i]))
        if e > max_error
            max_error = e
        end
    end
    @printf("   Max output error is %d\n\n", max_error)

    println("   Writing RAW Image")
    open(output_path, "w") do io
        write(io, output_image)
    end
    @printf("   Bytes written = %d\n\n", length(output_image) * 2)

    println(max_error <= 1 ? "PASS" : "FAIL")
    return 0
end

main()

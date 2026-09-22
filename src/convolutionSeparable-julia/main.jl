using CUDA
using Printf

const KERNEL_RADIUS = 8
const KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

function usage()
    println("Usage: ", PROGRAM_FILE, " <image width> <image height> <repeat>")
    exit(1)
end

function c_rand_sequence(n::Int)
    values = Vector{Cint}(undef, n)
    seed = Ref{Cuint}(2009)
    ccall(:srand, Cvoid, (Cuint,), seed[])
    @inbounds for i in 1:n
        values[i] = ccall(:rand, Cint, ())
    end
    return values
end

function init_inputs(image_w::Int, image_h::Int)
    raw = c_rand_sequence(KERNEL_LENGTH + image_w * image_h)
    kernel = Float32.(mod.(raw[1:KERNEL_LENGTH], 16))
    input = Float32.(mod.(raw[KERNEL_LENGTH + 1:end], 16))
    return kernel, input
end

function convolution_row_host(src, kernel, image_w::Int, image_h::Int)
    dst = zeros(Float32, image_w * image_h)
    @inbounds for y0 in 0:image_h-1, x0 in 0:image_w-1
        sumv = 0.0
        for k in -KERNEL_RADIUS:KERNEL_RADIUS
            d = x0 + k
            if 0 <= d < image_w
                sumv += Float64(src[y0 * image_w + d + 1]) * Float64(kernel[KERNEL_RADIUS - k + 1])
            end
        end
        dst[y0 * image_w + x0 + 1] = Float32(sumv)
    end
    return dst
end

function convolution_column_host(src, kernel, image_w::Int, image_h::Int)
    dst = zeros(Float32, image_w * image_h)
    @inbounds for y0 in 0:image_h-1, x0 in 0:image_w-1
        sumv = 0.0
        for k in -KERNEL_RADIUS:KERNEL_RADIUS
            d = y0 + k
            if 0 <= d < image_h
                sumv += Float64(src[d * image_w + x0 + 1]) * Float64(kernel[KERNEL_RADIUS - k + 1])
            end
        end
        dst[y0 * image_w + x0 + 1] = Float32(sumv)
    end
    return dst
end

function conv_rows_kernel!(dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = image_w * image_h
    if idx0 < total
        x0 = idx0 % image_w
        y0 = idx0 ÷ image_w
        sumv = 0.0f0
        k = Int32(-KERNEL_RADIUS)
        while k <= KERNEL_RADIUS
            d = x0 + k
            if d >= 0 && d < image_w
                sumv += @inbounds(src[y0 * image_w + d + Int32(1)]) *
                        @inbounds(kernel[Int32(KERNEL_RADIUS) - k + Int32(1)])
            end
            k += Int32(1)
        end
        @inbounds dst[idx0 + Int32(1)] = sumv
    end
    return
end

function conv_cols_kernel!(dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = image_w * image_h
    if idx0 < total
        x0 = idx0 % image_w
        y0 = idx0 ÷ image_w
        sumv = 0.0f0
        k = Int32(-KERNEL_RADIUS)
        while k <= KERNEL_RADIUS
            d = y0 + k
            if d >= 0 && d < image_h
                sumv += @inbounds(src[d * image_w + x0 + Int32(1)]) *
                        @inbounds(kernel[Int32(KERNEL_RADIUS) - k + Int32(1)])
            end
            k += Int32(1)
        end
        @inbounds dst[idx0 + Int32(1)] = sumv
    end
    return
end

function run_gpu!(d_output, d_buffer, d_input, d_kernel, image_w::Int, image_h::Int)
    total = image_w * image_h
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks conv_rows_kernel!(d_buffer, d_input, d_kernel, Int32(image_w), Int32(image_h))
    @cuda threads=threads blocks=blocks conv_cols_kernel!(d_output, d_buffer, d_kernel, Int32(image_w), Int32(image_h))
    return
end

function main()
    length(ARGS) == 3 || usage()
    image_w = parse(Int, ARGS[1])
    image_h = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    repeat > 0 || usage()

    kernel, input = init_inputs(image_w, image_h)
    d_kernel = CuArray(kernel)
    d_input = CuArray(input)
    d_buffer = CUDA.zeros(Float32, image_w * image_h)
    d_output = CUDA.zeros(Float32, image_w * image_h)

    run_gpu!(d_output, d_buffer, d_input, d_kernel, image_w, image_h)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        run_gpu!(d_output, d_buffer, d_input, d_kernel, image_w, image_h)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time %f (s)\n", elapsed * 1.0e-9 / repeat)

    output_gpu = Array(d_output)
    println("Comparing against Host/C++ computation...")
    buffer_cpu = convolution_row_host(input, kernel, image_w, image_h)
    output_cpu = convolution_column_host(buffer_cpu, kernel, image_w, image_h)
    delta = 0.0
    sumv = 0.0
    @inbounds for i in eachindex(output_cpu)
        diff = Float64(output_cpu[i] - output_gpu[i])
        delta += diff * diff
        sumv += Float64(output_cpu[i]) * Float64(output_cpu[i])
    end
    l2norm = sqrt(delta / sumv)
    @printf("Relative L2 norm: %.3e\n\n", l2norm)
    println(l2norm < 1.0e-6 ? "PASS" : "FAIL")
end

main()

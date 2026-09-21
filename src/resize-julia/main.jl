using CUDA
using Printf

# Julia port of resize-cuda.  Nearest and bilinear resize kernels preserve the
# CUDA indexing scheme with CHANNELS_PER_ITER fixed at 8.

const THREADS = 256
const GRID = 29184
const CHANNELS_PER_ITER = 8

function resize_kernel!(output, output_size::Int32, out_height::Int32, out_width::Int32,
                        input, in_height::Int32, in_width::Int32,
                        o2i_fy::Float32, o2i_fx::Float32,
                        do_round::Bool, half_pixel_centers::Bool)
    in_image_size = in_height * in_width
    out_image_size = out_height * out_width
    num_effective_channels = output_size ÷ out_image_size
    num_channel_iters_per_xy = num_effective_channels ÷ Int32(CHANNELS_PER_ITER)
    iters_required = num_channel_iters_per_xy * out_image_size
    iter = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while iter < iters_required
        c_start = (iter ÷ out_image_size) * Int32(CHANNELS_PER_ITER)
        y = (iter % out_image_size) ÷ out_width
        x = iter % out_width
        in_yf = half_pixel_centers ? (Float32(y) + 0.5f0) * o2i_fy : Float32(y) * o2i_fy
        in_xf = half_pixel_centers ? (Float32(x) + 0.5f0) * o2i_fx : Float32(x) * o2i_fx
        in_y = do_round ? Int32(round(in_yf)) : Int32(trunc(in_yf))
        in_x = do_round ? Int32(round(in_xf)) : Int32(trunc(in_xf))
        in_x = min(in_x, in_width - Int32(1))
        in_y = min(in_y, in_height - Int32(1))
        in_idx = c_start * in_image_size + in_y * in_width + in_x
        out_idx = c_start * out_image_size + y * out_width + x
        j = Int32(0)
        while j < Int32(CHANNELS_PER_ITER)
            @inbounds output[out_idx + Int32(1)] = input[in_idx + Int32(1)]
            in_idx += in_image_size
            out_idx += out_image_size
            j += Int32(1)
        end
        iter += stride
    end
    return
end

function resize_bilinear_kernel!(output, output_size::Int32, out_height::Int32, out_width::Int32,
                                 input, in_height::Int32, in_width::Int32,
                                 o2i_fy::Float32, o2i_fx::Float32, half_pixel_centers::Bool)
    in_image_size = in_height * in_width
    out_image_size = out_height * out_width
    num_effective_channels = output_size ÷ out_image_size
    num_channel_iters_per_xy = num_effective_channels ÷ Int32(CHANNELS_PER_ITER)
    iters_required = num_channel_iters_per_xy * out_image_size
    iter = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while iter < iters_required
        c_start = (iter ÷ out_image_size) * Int32(CHANNELS_PER_ITER)
        c_end = c_start + Int32(CHANNELS_PER_ITER)
        y = (iter % out_image_size) ÷ out_width
        x = iter % out_width
        in_x = half_pixel_centers ? max((Float32(x) + 0.5f0) * o2i_fx - 0.5f0, 0.0f0) : Float32(x) * o2i_fx
        in_y = half_pixel_centers ? max((Float32(y) + 0.5f0) * o2i_fy - 0.5f0, 0.0f0) : Float32(y) * o2i_fy
        in_x0 = Int32(trunc(in_x))
        in_x1 = min(in_x0 + Int32(1), in_width - Int32(1))
        in_y0 = Int32(trunc(in_y))
        in_y1 = min(in_y0, in_height - Int32(1))
        in_y2 = min(in_y0 + Int32(1), in_height - Int32(1))
        in_offset_r0 = c_start * in_image_size + in_y1 * in_width
        in_offset_r1 = c_start * in_image_size + in_y2 * in_width
        out_idx = c_start * out_image_size + y * out_width + x
        c = c_start
        while c < c_end
            v00 = Float32(@inbounds input[in_offset_r0 + in_x0 + Int32(1)])
            v01 = Float32(@inbounds input[in_offset_r0 + in_x1 + Int32(1)])
            v10 = Float32(@inbounds input[in_offset_r1 + in_x0 + Int32(1)])
            v11 = Float32(@inbounds input[in_offset_r1 + in_x1 + Int32(1)])
            value = v00 + (in_y - Float32(in_y0)) * (v10 - v00) +
                    (in_x - Float32(in_x0)) * (v01 - v00) +
                    (in_y - Float32(in_y0)) * (in_x - Float32(in_x0)) * (v11 - v01 - v10 + v00)
            @inbounds output[out_idx + Int32(1)] = unsafe_trunc(eltype(output), value)
            in_offset_r0 += in_image_size
            in_offset_r1 += in_image_size
            out_idx += out_image_size
            c += Int32(1)
        end
        iter += stride
    end
    return
end

function fill_input(::Type{T}, n::Int) where {T}
    data = Vector{T}(undef, n)
    for i in 1:n
        data[i] = T(i % 13)
    end
    return data
end

function reference_resize(::Type{T}, input, in_width, in_height, out_width, out_height, num_channels; bilinear=false) where {T}
    in_image_size = in_height * in_width
    out_image_size = out_height * out_width
    output = zeros(T, num_channels * out_image_size)
    fx = Float32(in_width / out_width)
    fy = Float32(in_height / out_height)
    for c_start in 0:CHANNELS_PER_ITER:num_channels-1
        for y in 0:out_height-1, x in 0:out_width-1
            if bilinear
                in_x = max((Float32(x) + 0.5f0) * fx - 0.5f0, 0.0f0)
                in_y = max((Float32(y) + 0.5f0) * fy - 0.5f0, 0.0f0)
                in_x0 = trunc(Int, in_x)
                in_x1 = min(in_x0 + 1, in_width - 1)
                in_y0 = trunc(Int, in_y)
                in_y1 = min(in_y0, in_height - 1)
                in_y2 = min(in_y0 + 1, in_height - 1)
                for j in 0:CHANNELS_PER_ITER-1
                    c = c_start + j
                    r0 = c * in_image_size + in_y1 * in_width
                    r1 = c * in_image_size + in_y2 * in_width
                    v00 = Float32(input[r0 + in_x0 + 1])
                    v01 = Float32(input[r0 + in_x1 + 1])
                    v10 = Float32(input[r1 + in_x0 + 1])
                    v11 = Float32(input[r1 + in_x1 + 1])
                    value = v00 + (in_y - Float32(in_y0)) * (v10 - v00) +
                            (in_x - Float32(in_x0)) * (v01 - v00) +
                            (in_y - Float32(in_y0)) * (in_x - Float32(in_x0)) * (v11 - v01 - v10 + v00)
                    output[c * out_image_size + y * out_width + x + 1] = unsafe_trunc(T, value)
                end
            else
                in_y = min(round(Int, (Float32(y) + 0.5f0) * fy), in_height - 1)
                in_x = min(round(Int, (Float32(x) + 0.5f0) * fx), in_width - 1)
                for j in 0:CHANNELS_PER_ITER-1
                    c = c_start + j
                    output[c * out_image_size + y * out_width + x + 1] =
                        input[c * in_image_size + in_y * in_width + in_x + 1]
                end
            end
        end
    end
    return output
end

function resize_image(::Type{T}, in_width, in_height, out_width, out_height, num_channels, repeat_n; bilinear=false) where {T}
    in_size = num_channels * in_height * in_width
    out_size = num_channels * out_height * out_width
    h_input = fill_input(T, in_size)
    d_input = CuArray(h_input)
    d_output = CUDA.zeros(T, out_size)
    fx = Float32(in_width / out_width)
    fy = Float32(in_height / out_height)

    CUDA.synchronize()
    t0 = time_ns()
    if bilinear
        for _ in 1:repeat_n
            @cuda threads=THREADS blocks=GRID resize_bilinear_kernel!(
                d_output, Int32(out_size), Int32(out_height), Int32(out_width),
                d_input, Int32(in_height), Int32(in_width), fy, fx, true)
        end
    else
        for _ in 1:repeat_n
            @cuda threads=THREADS blocks=GRID resize_kernel!(
                d_output, Int32(out_size), Int32(out_height), Int32(out_width),
                d_input, Int32(in_height), Int32(in_width), fy, fx, true, true)
        end
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - t0
    bytes = sizeof(T) * (in_size + out_size)
    @printf("Average kernel execution time: %lf (us)    Perf: %lf (GB/s)\n",
            elapsed_ns * 1e-3 / repeat_n, bytes * repeat_n / elapsed_ns)

    got = Array(d_output)
    ref = reference_resize(T, h_input, in_width, in_height, out_width, out_height, num_channels; bilinear=bilinear)
    return got == ref
end

function run_type(::Type{T}, bytes_label, in_width, in_height, out_width, out_height, num_channels, repeat_n) where {T}
    @printf("\nThe size of each pixel is %d byte%s\n", bytes_label, bytes_label == 1 ? "" : "s")
    ok1 = resize_image(T, in_width, in_height, out_width, out_height, num_channels, repeat_n)
    println("\nBilinear resizing")
    ok2 = resize_image(T, in_width, in_height, out_width, out_height, num_channels, repeat_n; bilinear=true)
    return ok1 && ok2
end

function main()
    if length(ARGS) != 6
        println("Usage: main.jl <input image width> <input image height>")
        println("          <output image width> <output image height>")
        println("          <image channels> <repeat>")
        return 1
    end
    in_width = parse(Int, ARGS[1])
    in_height = parse(Int, ARGS[2])
    out_width = parse(Int, ARGS[3])
    out_height = parse(Int, ARGS[4])
    num_channels = parse(Int, ARGS[5])
    repeat_n = parse(Int, ARGS[6])

    @printf("Resize %d images from (%d x %d) to (%d x %d)\n",
            num_channels, in_width, in_height, out_width, out_height)
    ok = run_type(UInt8, 1, in_width, in_height, out_width, out_height, num_channels, repeat_n)
    ok &= run_type(UInt16, 2, in_width, in_height, out_width, out_height, num_channels, repeat_n)
    ok &= run_type(UInt32, 4, in_width, in_height, out_width, out_height, num_channels, repeat_n)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())

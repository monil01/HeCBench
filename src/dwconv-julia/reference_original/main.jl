using CUDA
using Printf

# Julia port of dwconv-cuda.  The kernel uses flat NCHW indexing equivalent to
# the CUDA PackedTensorAccessor-based implementation.

const THREADS = 256

function dwconv_kernel!(input, output, weight, bias,
                        total::Int32, output_channels::Int32, depthwise_multiplier::Int32,
                        input_width::Int32, input_height::Int32,
                        output_width::Int32, output_height::Int32,
                        kernel_width::Int32, kernel_height::Int32,
                        stride_width::Int32, stride_height::Int32,
                        pad_width::Int32, pad_height::Int32,
                        dilation_width::Int32, dilation_height::Int32)
    linear = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if linear >= total
        return
    end

    tmp1 = linear ÷ output_width
    w = linear - tmp1 * output_width
    tmp2 = tmp1 ÷ output_height
    h = tmp1 - tmp2 * output_height
    tmp1 = tmp2
    tmp2 = tmp1 ÷ output_channels
    c = tmp1 - tmp2 * output_channels
    n = tmp2

    input_channel = c
    input_channels = output_channels
    if depthwise_multiplier != Int32(1)
        input_channel = input_channel ÷ depthwise_multiplier
        input_channels = input_channels ÷ depthwise_multiplier
    end

    weight_offset = c * kernel_height * kernel_width
    value = @inbounds bias[c + Int32(1)]
    offset0 = (n * input_channels + input_channel) * input_height * input_width
    k_h = Int32(0)
    while k_h < kernel_height
        k_w = Int32(0)
        while k_w < kernel_width
            h_in = -pad_height + h * stride_height + k_h * dilation_height
            w_in = -pad_width + w * stride_width + k_w * dilation_width
            if h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width
                offset = offset0 + h_in * input_width + w_in
                @inbounds value += weight[weight_offset + Int32(1)] * input[offset + Int32(1)]
            end
            weight_offset += Int32(1)
            k_w += Int32(1)
        end
        k_h += Int32(1)
    end
    @inbounds output[linear + Int32(1)] = value
    return
end

function fill_rand(n::Int)
    data = Vector{Float32}(undef, n)
    state = UInt64(123)
    for i in 1:n
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        data[i] = Float32((state >> 40) & UInt64(0xffffff)) / Float32(1 << 24)
    end
    return data
end

function dwconv2d_forward(m::Int, n::Int, input_channels::Int, h::Int, w::Int,
                          k_h::Int, k_w::Int, repeat_n::Int)
    output_channels = input_channels * m
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    output_h = (h + 2 * pad_h - ((k_h - 1) * dilation_h + 1)) ÷ stride_h + 1
    output_w = (w + 2 * pad_w - ((k_w - 1) * dilation_w + 1)) ÷ stride_w + 1

    input_size = n * input_channels * h * w
    weight_size = output_channels * k_h * k_w
    bias_size = output_channels
    output_size = n * output_channels * output_h * output_w

    h_input = fill_rand(input_size)
    h_weight = fill_rand(weight_size)
    h_bias = fill_rand(bias_size)
    d_input = CuArray(h_input)
    d_weight = CuArray(h_weight)
    d_bias = CuArray(h_bias)
    d_output = CUDA.zeros(Float32, output_size)
    blocks = cld(output_size, THREADS)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks dwconv_kernel!(
            d_input, d_output, d_weight, d_bias,
            Int32(output_size), Int32(output_channels), Int32(m),
            Int32(w), Int32(h), Int32(output_w), Int32(output_h),
            Int32(k_w), Int32(k_h), Int32(stride_w), Int32(stride_h),
            Int32(pad_w), Int32(pad_h), Int32(dilation_w), Int32(dilation_h))
    end
    CUDA.synchronize()
    @printf("Average execution time of dwconv2d_forward kernel: %f (ms)\n",
            (time_ns() - t0) * 1e-6 / repeat_n)

    out = Array(d_output)
    checksum = sum(out) / length(out)
    @printf("Checksum = %f\n", checksum)
    return isfinite(checksum)
end

function main()
    if length(ARGS) != 5
        println("Usage: main.jl <batch size> <number of input channels> <input height> <input width> <repeat>")
        return 1
    end
    n = parse(Int, ARGS[1])
    c = parse(Int, ARGS[2])
    h = parse(Int, ARGS[3])
    w = parse(Int, ARGS[4])
    repeat_n = parse(Int, ARGS[5])

    ok = true
    for m in 1:4
        for k in (1, 3, 5)
            @printf("batch = %d, input channel = %d, height = %d, width = %d, ", n, c, h, w)
            @printf("kernel size = %d, output channel = %d\n", k, m * c)
            ok &= dwconv2d_forward(m, n, c, h, w, k, k, repeat_n)
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())

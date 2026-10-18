using CUDA
using Printf
using Random

const THREADS_PER_BLOCK = 512

function vol2col_kernel!(range::Int64, data_vol, depth::Int32, height::Int32, width::Int32,
                         ksize_t::Int32, ksize_h::Int32, ksize_w::Int32,
                         pad_t::Int32, pad_h::Int32, pad_w::Int32,
                         stride_t::Int32, stride_h::Int32, stride_w::Int32,
                         dilation_t::Int32, dilation_h::Int32, dilation_w::Int32,
                         depth_col::Int32, height_col::Int32, width_col::Int32,
                         data_col)
    n = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = Int64(blockDim().x * gridDim().x)
    while n < range
        w_out = Int32(n % Int64(width_col))
        index = n ÷ Int64(width_col)
        h_out = Int32(index % Int64(height_col))
        index ÷= Int64(height_col)
        t_out = Int32(index % Int64(depth_col))
        channel_in = Int32(index ÷ Int64(depth_col))
        channel_out = channel_in * ksize_t * ksize_h * ksize_w
        t_in = t_out * stride_t - pad_t
        h_in = h_out * stride_h - pad_h
        w_in = w_out * stride_w - pad_w
        c0 = ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out
        @inbounds for i in Int32(0):(ksize_t - Int32(1))
            for j in Int32(0):(ksize_h - Int32(1))
                for k in Int32(0):(ksize_w - Int32(1))
                    t = t_in + i * dilation_t
                    h = h_in + j * dilation_h
                    w = w_in + k * dilation_w
                    v = if t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width
                        data_vol[((channel_in * depth + t) * height + h) * width + w + Int32(1)]
                    else
                        0.0f0
                    end
                    data_col[c0 + ((i * ksize_h + j) * ksize_w + k) *
                             depth_col * height_col * width_col + Int32(1)] = v
                end
            end
        end
        n += stride
    end
    return
end

function col2vol_kernel!(n::Int64, data_col, depth::Int32, height::Int32, width::Int32,
                         kernel_t::Int32, kernel_h::Int32, kernel_w::Int32,
                         pad_t::Int32, pad_h::Int32, pad_w::Int32,
                         stride_t::Int32, stride_h::Int32, stride_w::Int32,
                         dilation_t::Int32, dilation_h::Int32, dilation_w::Int32,
                         depth_col::Int32, height_col::Int32, width_col::Int32,
                         data_vol)
    index = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    grid_stride = Int64(blockDim().x * gridDim().x)
    while index < n
        w_im = Int32(index % Int64(width)) + pad_w
        h_im = Int32((index ÷ Int64(width)) % Int64(height)) + pad_h
        t_im = Int32((index ÷ Int64(width * height)) % Int64(depth)) + pad_t
        c_im = Int32(index ÷ Int64(width * height * depth))
        kernel_extent_w = (kernel_w - Int32(1)) * dilation_w + Int32(1)
        kernel_extent_h = (kernel_h - Int32(1)) * dilation_h + Int32(1)
        kernel_extent_t = (kernel_t - Int32(1)) * dilation_t + Int32(1)
        w_col_start = w_im < kernel_extent_w ? Int32(0) : (w_im - kernel_extent_w) ÷ stride_w + Int32(1)
        w_col_end = min(w_im ÷ stride_w + Int32(1), width_col)
        h_col_start = h_im < kernel_extent_h ? Int32(0) : (h_im - kernel_extent_h) ÷ stride_h + Int32(1)
        h_col_end = min(h_im ÷ stride_h + Int32(1), height_col)
        t_col_start = t_im < kernel_extent_t ? Int32(0) : (t_im - kernel_extent_t) ÷ stride_t + Int32(1)
        t_col_end = min(t_im ÷ stride_t + Int32(1), depth_col)
        val = 0.0f0
        @inbounds for t_col in t_col_start:(t_col_end - Int32(1))
            for h_col in h_col_start:(h_col_end - Int32(1))
                for w_col in w_col_start:(w_col_end - Int32(1))
                    t_k = t_im - t_col * stride_t
                    h_k = h_im - h_col * stride_h
                    w_k = w_im - w_col * stride_w
                    if t_k % dilation_t == 0 && h_k % dilation_h == 0 && w_k % dilation_w == 0
                        t_k ÷= dilation_t
                        h_k ÷= dilation_h
                        w_k ÷= dilation_w
                        idx_k = ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k
                        data_col_index = ((idx_k * depth_col + t_col) * height_col + h_col) *
                                         width_col + w_col
                        val += data_col[data_col_index + Int32(1)]
                    end
                end
            end
        end
        data_vol[index + Int64(1)] = val
        index += grid_stride
    end
    return
end

function vol2col_reference!(data_col, data_vol, channels, depth, height, width,
                            kt, kh, kw, pt, ph, pw, st, sh, sw, dt, dh, dw,
                            depth_col, height_col, width_col)
    fill!(data_col, 0.0f0)
    @inbounds for channel_in in 0:channels-1, t_out in 0:depth_col-1,
                  h_out in 0:height_col-1, w_out in 0:width_col-1
        channel_out = channel_in * kt * kh * kw
        t_in = t_out * st - pt
        h_in = h_out * sh - ph
        w_in = w_out * sw - pw
        c0 = ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out
        for i in 0:kt-1, j in 0:kh-1, k in 0:kw-1
            t = t_in + i * dt
            h = h_in + j * dh
            w = w_in + k * dw
            data_col[c0 + ((i * kh + j) * kw + k) * depth_col * height_col * width_col + 1] =
                (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width) ?
                data_vol[((channel_in * depth + t) * height + h) * width + w + 1] : 0.0f0
        end
    end
    return data_col
end

function col2vol_reference!(data_vol, data_col, channels, depth, height, width,
                            kt, kh, kw, pt, ph, pw, st, sh, sw, dt, dh, dw,
                            depth_col, height_col, width_col)
    @inbounds for channel_in in 0:channels-1, t_out in 0:depth-1,
                  h_out in 0:height-1, w_out in 0:width-1
        val = 0.0f0
        w_im = w_out + pw
        h_im = h_out + ph
        t_im = t_out + pt
        kernel_extent_w = (kw - 1) * dw + 1
        kernel_extent_h = (kh - 1) * dh + 1
        kernel_extent_t = (kt - 1) * dt + 1
        w_col_start = w_im < kernel_extent_w ? 0 : (w_im - kernel_extent_w) ÷ sw + 1
        w_col_end = min(w_im ÷ sw + 1, width_col)
        h_col_start = h_im < kernel_extent_h ? 0 : (h_im - kernel_extent_h) ÷ sh + 1
        h_col_end = min(h_im ÷ sh + 1, height_col)
        t_col_start = t_im < kernel_extent_t ? 0 : (t_im - kernel_extent_t) ÷ st + 1
        t_col_end = min(t_im ÷ st + 1, depth_col)
        for t_col in t_col_start:t_col_end-1, h_col in h_col_start:h_col_end-1,
            w_col in w_col_start:w_col_end-1
            t_k = t_im - t_col * st
            h_k = h_im - h_col * sh
            w_k = w_im - w_col * sw
            if t_k % dt == 0 && h_k % dh == 0 && w_k % dw == 0
                t_k ÷= dt
                h_k ÷= dh
                w_k ÷= dw
                idx_k = ((channel_in * kt + t_k) * kh + h_k) * kw + w_k
                data_col_index = ((idx_k * depth_col + t_col) * height_col + h_col) *
                                 width_col + w_col
                val += data_col[data_col_index + 1]
            end
        end
        data_vol[channel_in * width * height * depth + t_out * width * height +
                 h_out * width + w_out + 1] = val
    end
    return data_vol
end

function run_case(repeat::Int, k::Int)
    channels, depth, height, width = 4, 3, 255, 255
    pad_t = pad_h = pad_w = 1
    stride_t = stride_h = stride_w = 2
    dilation_t = dilation_h = dilation_w = 2
    depth_col, height_col, width_col = 3, 255, 255
    ksize_t = ksize_h = ksize_w = k

    vol_size = channels * (2 * pad_t + depth) * (2 * pad_h + height) * (2 * pad_w + width)
    col_size = (channels * ksize_t * ksize_h * ksize_w + 1) *
               (depth_col + pad_t) * (height_col + pad_h) * (width_col + pad_w)
    rng = MersenneTwister(123)
    h_data_vol = rand(rng, Float32, vol_size)
    h_data_col_ref = zeros(Float32, col_size)
    h_data_vol_ref = copy(h_data_vol)

    d_data_vol = CuArray(h_data_vol)
    d_data_col = CUDA.zeros(Float32, col_size)
    n = Int64(channels) * depth_col * height_col * width_col
    blocks = cld(n, THREADS_PER_BLOCK)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS_PER_BLOCK blocks=blocks vol2col_kernel!(
            n, d_data_vol, Int32(depth), Int32(height), Int32(width),
            Int32(ksize_t), Int32(ksize_h), Int32(ksize_w),
            Int32(pad_t), Int32(pad_h), Int32(pad_w),
            Int32(stride_t), Int32(stride_h), Int32(stride_w),
            Int32(dilation_t), Int32(dilation_h), Int32(dilation_w),
            Int32(depth_col), Int32(height_col), Int32(width_col), d_data_col)
    end
    CUDA.synchronize()
    @printf("Average execution time of vol2col kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)

    h_data_col = Array(d_data_col)
    vol2col_reference!(h_data_col_ref, h_data_vol, channels, depth, height, width,
                       ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
                       stride_t, stride_h, stride_w, dilation_t, dilation_h,
                       dilation_w, depth_col, height_col, width_col)
    println(h_data_col == h_data_col_ref ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS_PER_BLOCK blocks=blocks col2vol_kernel!(
            n, d_data_col, Int32(depth), Int32(height), Int32(width),
            Int32(ksize_t), Int32(ksize_h), Int32(ksize_w),
            Int32(pad_t), Int32(pad_h), Int32(pad_w),
            Int32(stride_t), Int32(stride_h), Int32(stride_w),
            Int32(dilation_t), Int32(dilation_h), Int32(dilation_w),
            Int32(depth_col), Int32(height_col), Int32(width_col), d_data_vol)
    end
    CUDA.synchronize()
    @printf("Average execution time of col2vol kernel: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)

    h_data_vol_gpu = Array(d_data_vol)
    col2vol_reference!(h_data_vol_ref, h_data_col_ref, channels, depth, height, width,
                       ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
                       stride_t, stride_h, stride_w, dilation_t, dilation_h,
                       dilation_w, depth_col, height_col, width_col)
    ok = all(abs.(h_data_vol_ref .- h_data_vol_gpu) .<= 1.0f-3)
    println(ok ? "PASS" : "FAIL")
    return ok && h_data_col == h_data_col_ref
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    all_ok = true
    for k in 1:2:9
        @printf("\nkernel size: %d\n", k)
        all_ok &= run_case(repeat, k)
    end
    return all_ok ? 0 : 1
end

exit(main(ARGS))

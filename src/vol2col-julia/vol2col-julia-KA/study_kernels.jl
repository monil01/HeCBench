using KernelAbstractions
@kernel function vol2col_ka!(range::Int64, data_vol, depth::Int32, height::Int32, width::Int32,
                         ksize_t::Int32, ksize_h::Int32, ksize_w::Int32,
                         pad_t::Int32, pad_h::Int32, pad_w::Int32,
                         stride_t::Int32, stride_h::Int32, stride_w::Int32,
                         dilation_t::Int32, dilation_h::Int32, dilation_w::Int32,
                         depth_col::Int32, height_col::Int32, width_col::Int32,
                         data_col, study_grid_stride::Int64)
    g=@index(Global,Linear)
    n=Int64(g-1)
    stride=study_grid_stride
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
end

@kernel function col2vol_ka!(n::Int64, data_col, depth::Int32, height::Int32, width::Int32,
                         kernel_t::Int32, kernel_h::Int32, kernel_w::Int32,
                         pad_t::Int32, pad_h::Int32, pad_w::Int32,
                         stride_t::Int32, stride_h::Int32, stride_w::Int32,
                         dilation_t::Int32, dilation_h::Int32, dilation_w::Int32,
                         depth_col::Int32, height_col::Int32, width_col::Int32,
                         data_vol, study_grid_stride::Int64)
    g=@index(Global,Linear)
    index=Int64(g-1)
    grid_stride=study_grid_stride
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
end


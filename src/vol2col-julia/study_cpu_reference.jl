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


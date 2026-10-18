using CUDA
using Printf
using Random

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function upsample_forward_kernel!(x, out, B::UInt64, C::UInt64, H::UInt64, W::UInt64, total::UInt64)
    flat = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if flat >= total
        return
    end
    img_size = H * W
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    img_out_size = h_out * w_out
    b = flat ÷ (C * img_size)
    c = (flat ÷ img_size) % C
    i = (flat ÷ W) % H
    j = flat % W
    x_idx = b * C * img_size + c * img_size + i * W + j
    out_base = b * C * img_out_size + c * img_out_size + UInt64(2) * i * w_out + UInt64(2) * j
    @inbounds val = x[Int(x_idx) + 1]
    @inbounds out[Int(out_base) + 1] = val
    @inbounds out[Int(out_base) + 2] = val
    @inbounds out[Int(out_base + w_out) + 1] = val
    @inbounds out[Int(out_base + w_out) + 2] = val
    return
end

function upsample_forward_kernel2!(x, out, B::UInt64, C::UInt64, H::UInt64, W::UInt64)
    in_x = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    in_y = UInt64((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    bc = UInt64(blockIdx().z - Int32(1))
    if in_x >= W || in_y >= H
        return
    end
    b = bc ÷ C
    c = bc % C
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    @inbounds val = x[Int((b * C + c) * H * W + in_y * W + in_x) + 1]
    out_base = (b * C + c) * h_out * w_out + (in_y * UInt64(2)) * w_out + in_x * UInt64(2)
    @inbounds out[Int(out_base) + 1] = val
    @inbounds out[Int(out_base) + 2] = val
    @inbounds out[Int(out_base + w_out) + 1] = val
    @inbounds out[Int(out_base + w_out) + 2] = val
    return
end

function upsample_backward_kernel!(dout, dx, B::UInt64, C::UInt64, H::UInt64, W::UInt64, total::UInt64)
    flat = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if flat >= total
        return
    end
    img_size = H * W
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    img_out_size = h_out * w_out
    b = flat ÷ (C * img_size)
    c = (flat ÷ img_size) % C
    i = (flat ÷ W) % H
    j = flat % W
    out_base = b * C * img_out_size + c * img_out_size + UInt64(2) * i * w_out + UInt64(2) * j
    @inbounds dx[Int(flat) + 1] = dout[Int(out_base) + 1] + dout[Int(out_base) + 2] +
                                  dout[Int(out_base + w_out) + 1] + dout[Int(out_base + w_out) + 2]
    return
end

function upsample_backward_kernel2!(dout, dx, B::UInt64, C::UInt64, H::UInt64, W::UInt64)
    in_x = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    in_y = UInt64((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    bc = UInt64(blockIdx().z - Int32(1))
    if in_x >= W || in_y >= H
        return
    end
    b = bc ÷ C
    c = bc % C
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    out_base = b * C * h_out * w_out + c * h_out * w_out + UInt64(2) * in_y * w_out + UInt64(2) * in_x
    @inbounds dx[Int((b * C + c) * H * W + in_y * W + in_x) + 1] =
        dout[Int(out_base) + 1] + dout[Int(out_base) + 2] +
        dout[Int(out_base + w_out) + 1] + dout[Int(out_base + w_out) + 2]
    return
end


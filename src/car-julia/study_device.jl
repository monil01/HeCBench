
struct Params
    output_dim_b::Int32
    output_dim_c::Int32
    output_dim_h::Int32
    output_dim_w::Int32
    kernel_size::Int32
    image_w::Int32
    image_h::Int32
end

function lcg_random_double(seed::Base.RefValue{UInt64})
    seed[] = UInt64(2806196910506780709) * seed[] + UInt64(1)
    seed[] &= UInt64(0x7fffffffffffffff)
    return Float64(seed[]) / Float64(UInt64(1) << 63)
end

function idx_img(b, c, y, x, dim_c, img_w, img_h)
    return (b * dim_c + c) * img_w * img_h + y * img_w + x + 1
end

function idx_4d(b, c, y, x, cdim, dim_w, dim_h)
    return ((b * cdim + c) * dim_w * dim_h + y * dim_w + x) + 1
end

function car_lane!(gid, img, kernels, offsets_h, offsets_v, output, p, offset_unit::Int32,
                     padding::Int32, n::Int64)
    gid >= n && return

    dim_b = Int64(p.output_dim_b)
    dim_c = Int64(p.output_dim_c)
    dim_h = Int64(p.output_dim_h)
    dim_w = Int64(p.output_dim_w)
    kernels_size = Int64(p.kernel_size)
    img_w = Int64(p.image_w)
    img_h = Int64(p.image_h)

    vol_size = dim_c * dim_h * dim_w
    img_size = dim_h * dim_w
    idb = (gid ÷ vol_size) % dim_b
    idc = (gid ÷ img_size) % dim_c
    idy = (gid ÷ dim_w) % dim_h
    idx = gid % dim_w

    k_size = Int64(floor(sqrt(Float32(kernels_size))))
    w = img_w - Int64(2) * Int64(padding)
    h = img_h - Int64(2) * Int64(padding)

    result = Float32(0)
    for k_y in Int64(0):(k_size - Int64(1))
        for k_x in Int64(0):(k_size - Int64(1))
            kc = k_size * k_y + k_x
            offset_h = offsets_h[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)] * Float32(offset_unit)
            offset_v = offsets_v[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)] * Float32(offset_unit)

            p_x = (Float32(idx) + Float32(0.5)) / Float32(dim_w) * Float32(w) + Float32(k_x) + offset_h - Float32(0.5)
            p_y = (Float32(idy) + Float32(0.5)) / Float32(dim_h) * Float32(h) + Float32(k_y) + offset_v - Float32(0.5)
            x_floor = floor(p_x)
            y_floor = floor(p_y)
            alpha = p_x - x_floor
            beta = p_y - y_floor

            x_l = max(min(Int64(x_floor), w + Int64(2) * Int64(padding) - Int64(1)), Int64(0))
            x_r = max(min(x_l + Int64(1), w + Int64(2) * Int64(padding) - Int64(1)), Int64(0))
            y_t = max(min(Int64(y_floor), h + Int64(2) * Int64(padding) - Int64(1)), Int64(0))
            y_b = max(min(y_t + Int64(1), h + Int64(2) * Int64(padding) - Int64(1)), Int64(0))

            val = (Float32(1) - alpha) * (Float32(1) - beta) * img[idx_img(idb, idc, y_t, x_l, dim_c, img_w, img_h)]
            val += alpha * (Float32(1) - beta) * img[idx_img(idb, idc, y_t, x_r, dim_c, img_w, img_h)]
            val += (Float32(1) - alpha) * beta * img[idx_img(idb, idc, y_b, x_l, dim_c, img_w, img_h)]
            val += alpha * beta * img[idx_img(idb, idc, y_b, x_r, dim_c, img_w, img_h)]
            result += val * kernels[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)]
        end
    end
    output[gid + 1] = result
    return
end


# Source-derived box-Hessian device helper; LICENSE.native.
# Authored during KA translation; reused by JACC, not new reference preparation.
@inline function doh_clip(x::Int32, low::Int32, high::Int32)
    if x > high
        return high
    elseif x < low
        return low
    end
    return x
end

@inline function doh_box(img, img_rows::Int32, img_cols::Int32, r_in::Int32, c_in::Int32,
                       rl::Int32, cl::Int32)
    r = doh_clip(r_in, Int32(0), img_rows - Int32(1))
    c = doh_clip(c_in, Int32(0), img_cols - Int32(1))
    r2 = doh_clip(r + rl, Int32(0), img_rows - Int32(1))
    c2 = doh_clip(c + cl, Int32(0), img_cols - Int32(1))
    ans = @inbounds img[Int64(r) * Int64(img_cols) + Int64(c) + 1] +
          img[Int64(r2) * Int64(img_cols) + Int64(c2) + 1] -
          img[Int64(r) * Int64(img_cols) + Int64(c2) + 1] -
          img[Int64(r2) * Int64(img_cols) + Int64(c) + 1]
    return max(0.0f0, ans)
end

@inline function doh_cell!(img, img_rows::Int32, img_cols::Int32, sigma::Float32, out, tid0::Int32)
    total = img_rows * img_cols
    if tid0 >= total
        return
    end

    r = tid0 ÷ img_cols
    c = tid0 - r * img_cols
    size = Int32(trunc(3.0f0 * sigma))
    b = (size - Int32(1)) ÷ Int32(2) + Int32(1)
    l = size ÷ Int32(3)
    w = size
    w_i = 1.0f0 / Float32(size * size)

    tl = doh_box(img, img_rows, img_cols, r - l, c - l, l, l)
    br = doh_box(img, img_rows, img_cols, r + Int32(1), c + Int32(1), l, l)
    bl = doh_box(img, img_rows, img_cols, r - l, c + Int32(1), l, l)
    tr = doh_box(img, img_rows, img_cols, r + Int32(1), c - l, l, l)
    dxy = -(bl + tr - tl - br) * w_i

    mid = doh_box(img, img_rows, img_cols, r - l + Int32(1), c - l, Int32(2) * l - Int32(1), w)
    side = doh_box(img, img_rows, img_cols, r - l + Int32(1), c - l ÷ Int32(2), Int32(2) * l - Int32(1), l)
    dxx = -(mid - 3.0f0 * side) * w_i

    mid = doh_box(img, img_rows, img_cols, r - l, c - b + Int32(1), w, Int32(2) * b - Int32(1))
    side = doh_box(img, img_rows, img_cols, r - b ÷ Int32(2), c - b + Int32(1), b, Int32(2) * b - Int32(1))
    dyy = -(mid - 3.0f0 * side) * w_i

    @inbounds out[Int64(tid0) + 1] = dxx * dyy - 0.81f0 * (dxy * dxy)
    return
end


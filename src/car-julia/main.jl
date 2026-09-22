using CUDA
using Printf

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
    return ((b * dim_c + c) * img_w + y) * img_h + x + 1
end

function idx_4d(b, c, y, x, cdim, dim_w, dim_h)
    return ((b * cdim + c) * dim_w * dim_h + y * dim_w + x) + 1
end

function car_kernel!(img, kernels, offsets_h, offsets_v, output, p, offset_unit::Int32,
                     padding::Int32, n::Int64)
    gid = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
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

function reference!(img, kernels, offsets_h, offsets_v, output, p::Params, offset_unit::Int32, padding::Int32)
    dim_b = Int(p.output_dim_b)
    dim_c = Int(p.output_dim_c)
    dim_h = Int(p.output_dim_h)
    dim_w = Int(p.output_dim_w)
    k_size = Int(floor(sqrt(Float32(p.kernel_size))))
    img_w = Int(p.image_w)
    img_h = Int(p.image_h)
    w = img_w - 2 * Int(padding)
    h = img_h - 2 * Int(padding)

    for idb in 0:(dim_b - 1), idc in 0:(dim_c - 1), idy in 0:(dim_h - 1), idx in 0:(dim_w - 1)
        result = Float32(0)
        for k_y in 0:(k_size - 1), k_x in 0:(k_size - 1)
            kc = k_size * k_y + k_x
            off_h = offsets_h[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)] * Float32(offset_unit)
            off_v = offsets_v[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)] * Float32(offset_unit)
            p_x = (Float32(idx) + Float32(0.5)) / Float32(dim_w) * Float32(w) + Float32(k_x) + off_h - Float32(0.5)
            p_y = (Float32(idy) + Float32(0.5)) / Float32(dim_h) * Float32(h) + Float32(k_y) + off_v - Float32(0.5)
            x_floor = floor(p_x)
            y_floor = floor(p_y)
            alpha = p_x - x_floor
            beta = p_y - y_floor
            x_l = max(min(Int(x_floor), w + 2 * Int(padding) - 1), 0)
            x_r = max(min(x_l + 1, w + 2 * Int(padding) - 1), 0)
            y_t = max(min(Int(y_floor), h + 2 * Int(padding) - 1), 0)
            y_b = max(min(y_t + 1, h + 2 * Int(padding) - 1), 0)
            val = (Float32(1) - alpha) * (Float32(1) - beta) * img[idx_img(idb, idc, y_t, x_l, dim_c, img_w, img_h)]
            val += alpha * (Float32(1) - beta) * img[idx_img(idb, idc, y_t, x_r, dim_c, img_w, img_h)]
            val += (Float32(1) - alpha) * beta * img[idx_img(idb, idc, y_b, x_l, dim_c, img_w, img_h)]
            val += alpha * beta * img[idx_img(idb, idc, y_b, x_r, dim_c, img_w, img_h)]
            result += val * kernels[idx_4d(idb, kc, idy, idx, k_size * k_size, dim_w, dim_h)]
        end
        output[idx_4d(idb, idc, idy, idx, dim_c, dim_w, dim_h)] = result
    end
    return output
end

function main()
    length(ARGS) == 1 || error("Usage: julia main.jl <repeat>")
    repeat = parse(Int, ARGS[1])

    p = Params(128, 3, 480, 640, 9, 1024, 1024)
    padding = Int32(1)
    image_size = Int(p.output_dim_b) * Int(p.output_dim_c) * (Int(p.image_w) + Int(padding)) * (Int(p.image_h) + Int(padding))
    offset_size = Int(p.output_dim_b) * Int(p.kernel_size) * Int(p.output_dim_w) * Int(p.output_dim_h)
    kernel_size = offset_size
    output_size = Int(p.output_dim_b) * Int(p.output_dim_c) * Int(p.output_dim_w) * Int(p.output_dim_h)

    seed = Ref(UInt64(123))
    img = Vector{Float32}(undef, image_size)
    kernels = Vector{Float32}(undef, kernel_size)
    offsets_h = Vector{Float32}(undef, offset_size)
    offsets_v = Vector{Float32}(undef, offset_size)
    for i in eachindex(img)
        img[i] = Float32(UInt8(floor(256 * lcg_random_double(seed))))
    end
    for i in eachindex(kernels)
        kernels[i] = Float32(UInt8(floor(256 * lcg_random_double(seed))))
    end
    for i in eachindex(offsets_h)
        offsets_h[i] = Float32(lcg_random_double(seed))
        offsets_v[i] = Float32(lcg_random_double(seed))
    end

    d_img = CuArray(img)
    d_kernels = CuArray(kernels)
    d_offsets_h = CuArray(offsets_h)
    d_offsets_v = CuArray(offsets_v)
    d_output = CUDA.zeros(Float32, output_size)

    threads = 256
    blocks = cld(output_size, threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks car_kernel!(d_img, d_kernels, d_offsets_h, d_offsets_v,
                                                        d_output, p, Int32(1), padding, Int64(output_size))
    end
    CUDA.synchronize()
    dt_s = (time_ns() - t0) * 1e-9 / repeat
    @printf("Average kernel execution time %f (s)\n", dt_s)

    output_ref = Vector{Float32}(undef, output_size)
    reference!(img, kernels, offsets_h, offsets_v, output_ref, p, Int32(1), padding)
    output = Array(d_output)
    rmse = sqrt(sum((output_ref .- output) .^ 2) / output_size)
    @printf("RMSE: %f\n", rmse)
    println(rmse <= 1.0f-3 ? "PASS" : "FAIL")
end

main()

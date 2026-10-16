using CUDA
using Printf

function bilateral_filter_kernel!(input, output, w::Int32, h::Int32,
                                  a_square::Float32, variance_I::Float32,
                                  variance_spatial::Float32, radius::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    idy0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if idx0 >= w || idy0 >= h
        return
    end

    id = idy0 * w + idx0
    center = input[id + Int32(1)]
    res = Float32(0)
    normalization = Float32(0)

    oi = -radius
    while oi <= radius
        oj = -radius
        while oj <= radius
            idk = idx0 + oi
            idl = idy0 + oj
            if idk < 0
                idk = -idk
            end
            if idl < 0
                idl = -idl
            end
            if idk > w - Int32(1)
                idk = w - Int32(1) - oi
            end
            if idl > h - Int32(1)
                idl = h - Int32(1) - oj
            end

            id_w = idl * w + idk
            I_w = input[id_w + Int32(1)]
            range = -((center - I_w) * (center - I_w)) / (Float32(2) * variance_I)
            spatial = -(Float32((idk - idx0) * (idk - idx0) + (idl - idy0) * (idl - idy0))) /
                      (Float32(2) * variance_spatial)
            weight = a_square * exp(spatial + range)
            normalization += weight
            res += I_w * weight
            oj += Int32(1)
        end
        oi += Int32(1)
    end
    output[id + Int32(1)] = res / normalization
    return
end

function reference!(input, output, w::Int, h::Int, a_square::Float32,
                    variance_I::Float32, variance_spatial::Float32, radius::Int)
    for idx0 in 0:w-1, idy0 in 0:h-1
        id = idy0 * w + idx0 + 1
        center = input[id]
        res = Float32(0)
        normalization = Float32(0)
        for oi in -radius:radius, oj in -radius:radius
            idk = idx0 + oi
            idl = idy0 + oj
            if idk < 0
                idk = -idk
            end
            if idl < 0
                idl = -idl
            end
            if idk > w - 1
                idk = w - 1 - oi
            end
            if idl > h - 1
                idl = h - 1 - oj
            end
            id_w = idl * w + idk + 1
            I_w = input[id_w]
            range = -((center - I_w) * (center - I_w)) / (Float32(2) * variance_I)
            spatial = -(Float32((idk - idx0)^2 + (idl - idy0)^2)) / (Float32(2) * variance_spatial)
            weight = a_square * exp(spatial + range)
            normalization += weight
            res += I_w * weight
        end
        output[id] = res / normalization
    end
    return output
end

function c_rand_mod(n::Int)
    return Int(ccall(:rand, Cint, ())) % n
end

function run_radius!(d_src, d_dst, h_src, h_dst, r_dst, w::Int, h::Int,
                     a_square::Float32, variance_I::Float32,
                     variance_spatial::Float32, repeat::Int, radius::Int)
    threads = (16, 16)
    blocks = (cld(w, 16), cld(h, 16))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks bilateral_filter_kernel!(
            d_src, d_dst, Int32(w), Int32(h), a_square, variance_I,
            variance_spatial, Int32(radius))
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - t0
    @printf("Average kernel execution time (%dx%d) %f (ms)\n",
            radius, radius, (elapsed_ns * 1e-6) / repeat)

    copyto!(h_dst, d_dst)
    reference!(h_src, r_dst, w, h, a_square, variance_I, variance_spatial, radius)
    ok = true
    for i in eachindex(h_dst)
        if abs(r_dst[i] - h_dst[i]) > 1f-3
            ok = false
            break
        end
    end
    return ok
end

function main()
    if length(ARGS) != 5
        println("Usage: main.jl <image width> <image height> <intensity> <spatial> <repeat>")
        exit(1)
    end
    w = parse(Int, ARGS[1])
    h = parse(Int, ARGS[2])
    variance_I = Float32(parse(Float64, ARGS[3]))
    variance_spatial = Float32(parse(Float64, ARGS[4]))
    repeat = parse(Int, ARGS[5])
    img_size = w * h
    a_square = Float32(0.5) / (variance_I * Float32(pi))

    ccall(:srand, Cvoid, (Cuint,), Cuint(123))
    h_src = Vector{Float32}(undef, img_size)
    for i in eachindex(h_src)
        h_src[i] = Float32(c_rand_mod(256))
    end
    h_dst = Vector{Float32}(undef, img_size)
    r_dst = Vector{Float32}(undef, img_size)
    d_src = CuArray(h_src)
    d_dst = CuArray{Float32}(undef, img_size)

    ok = true
    ok &= run_radius!(d_src, d_dst, h_src, h_dst, r_dst, w, h, a_square,
                      variance_I, variance_spatial, repeat, 3)
    ok &= run_radius!(d_src, d_dst, h_src, h_dst, r_dst, w, h, a_square,
                      variance_I, variance_spatial, repeat, 6)
    ok &= run_radius!(d_src, d_dst, h_src, h_dst, r_dst, w, h, a_square,
                      variance_I, variance_spatial, repeat, 9)
    println(ok ? "PASS" : "FAIL")
    ok || exit(1)
end

main()

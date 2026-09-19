using CUDA
using Printf
using Random

@inline function to_u8(x::Float32)
    return UInt8(trunc(Int32, x) % Int32(256))
end

function find_moving_pixels_kernel!(img_size::Int32, img, img1, img2, tn, mp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= img_size
        return
    end
    idx = i + Int32(1)
    @inbounds diff1 = abs(Int32(img[idx]) - Int32(img1[idx]))
    @inbounds diff2 = abs(Int32(img[idx]) - Int32(img2[idx]))
    @inbounds threshold = Int32(tn[idx])
    @inbounds mp[idx] = (diff1 > threshold || diff2 > threshold) ? UInt8(255) : UInt8(0)
    return
end

function update_background_kernel!(img_size::Int32, img, mp, bn)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= img_size
        return
    end
    idx = i + Int32(1)
    @inbounds if mp[idx] == UInt8(0)
        bn[idx] = to_u8(0.92f0 * Float32(bn[idx]) + 0.08f0 * Float32(img[idx]))
    end
    return
end

function update_threshold_kernel!(img_size::Int32, img, mp, bn, tn)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= img_size
        return
    end
    idx = i + Int32(1)
    @inbounds if mp[idx] == UInt8(0)
        th = 0.92f0 * Float32(tn[idx]) + 0.24f0 * Float32(Int32(img[idx]) - Int32(bn[idx]))
        tn[idx] = to_u8(max(th, 20.0f0))
    end
    return
end

function merge_kernel!(img_size::Int32, img, img1, img2, tn, bn)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= img_size
        return
    end
    idx = i + Int32(1)
    @inbounds diff1 = abs(Int32(img[idx]) - Int32(img1[idx]))
    @inbounds diff2 = abs(Int32(img[idx]) - Int32(img2[idx]))
    @inbounds threshold = Int32(tn[idx])
    if diff1 <= threshold && diff2 <= threshold
        @inbounds bn[idx] = to_u8(0.92f0 * Float32(bn[idx]) + 0.08f0 * Float32(img[idx]))
        @inbounds th = 0.92f0 * Float32(tn[idx]) + 0.24f0 * Float32(Int32(img[idx]) - Int32(bn[idx]))
        @inbounds tn[idx] = to_u8(max(th, 20.0f0))
    end
    return
end

function merge_ref!(img, img1, img2, tn, bn)
    @inbounds for i in eachindex(img)
        if abs(Int(img[i]) - Int(img1[i])) <= Int(tn[i]) &&
           abs(Int(img[i]) - Int(img2[i])) <= Int(tn[i])
            bn[i] = UInt8(mod(trunc(Int, 0.92f0 * Float32(bn[i]) + 0.08f0 * Float32(img[i])), 256))
            th = 0.92f0 * Float32(tn[i]) + 0.24f0 * Float32(Int(img[i]) - Int(bn[i]))
            tn[i] = UInt8(mod(trunc(Int, max(th, 20.0f0)), 256))
        end
    end
end

function fill_random_frame!(rng::MersenneTwister, frame::Vector{UInt8})
    @inbounds for i in eachindex(frame)
        frame[i] = UInt8(rand(rng, 0:255))
    end
end

function run_background(width::Int, height::Int, merged::Int, repeat::Int)
    img_size = width * height
    rng = MersenneTwister(123)

    img = Vector{UInt8}(undef, img_size)
    img1 = Vector{UInt8}(undef, img_size)
    img2 = Vector{UInt8}(undef, img_size)
    bn = Vector{UInt8}(undef, img_size)
    bn_ref = Vector{UInt8}(undef, img_size)
    tn = fill(UInt8(128), img_size)
    tn_ref = fill(UInt8(128), img_size)

    @inbounds for i in 1:img_size
        v = UInt8(rand(rng, 0:255))
        bn[i] = v
        bn_ref[i] = v
    end

    d_img = CUDA.zeros(UInt8, img_size)
    d_img1 = CUDA.zeros(UInt8, img_size)
    d_img2 = CUDA.zeros(UInt8, img_size)
    d_bn = CuArray(bn)
    d_tn = CuArray(tn)
    d_mp = CUDA.zeros(UInt8, img_size)

    threads = 256
    blocks = cld(img_size, threads)
    elapsed_ns = 0

    for i in 0:repeat-1
        fill_random_frame!(rng, img)
        copyto!(d_img, img)

        d_img2, d_img1, d_img = d_img1, d_img, d_img2
        img2, img1, img = img1, img, img2

        if i >= 2
            CUDA.synchronize()
            start = time_ns()
            if merged != 0
                @cuda threads=threads blocks=blocks merge_kernel!(
                    Int32(img_size), d_img, d_img1, d_img2, d_tn, d_bn)
            else
                @cuda threads=threads blocks=blocks find_moving_pixels_kernel!(
                    Int32(img_size), d_img, d_img1, d_img2, d_tn, d_mp)
                @cuda threads=threads blocks=blocks update_background_kernel!(
                    Int32(img_size), d_img, d_mp, d_bn)
                @cuda threads=threads blocks=blocks update_threshold_kernel!(
                    Int32(img_size), d_img, d_mp, d_bn, d_tn)
            end
            CUDA.synchronize()
            elapsed_ns += time_ns() - start
            merge_ref!(img, img1, img2, tn_ref, bn_ref)
        end
    end

    kernel_time_us = repeat <= 2 ? 0.0 : elapsed_ns * 1e-3 / (repeat - 2)
    @printf("Average kernel execution time: %f (us)\n", kernel_time_us)

    copyto!(tn, d_tn)
    copyto!(bn, d_bn)
    max_error = 0
    @inbounds for i in eachindex(tn)
        e = abs(Int(tn[i]) - Int(tn_ref[i]))
        if e > max_error
            max_error = e
        end
    end
    @inbounds for i in eachindex(bn)
        e = abs(Int(bn[i]) - Int(bn_ref[i]))
        if e > max_error
            max_error = e
        end
    end
    @printf("Max error is %d\n", max_error)
    println(max_error == 0 ? "PASS" : "FAIL")
    return max_error == 0 ? 0 : 1
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <image width> <image height> <merge> <repeat>")
        return 1
    end
    return run_background(parse(Int, ARGS[1]), parse(Int, ARGS[2]),
                          parse(Int, ARGS[3]), parse(Int, ARGS[4]))
end

exit(main())

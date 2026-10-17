using CUDA
using Printf

const THREADS_X = 16
const THREADS_Y = 16

@inline function linidx0(x::Int32, y::Int32, lx::Int32)
    return y * lx + x + Int32(1)
end

function smoothing_filter_kernel!(
    lx::Int32,
    ly::Int32,
    threshold::Int32,
    max_rad::Int32,
    img,
    box,
    norm,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        sum = 0.0f0
        q = Int32(1)
        s = Int32(1)
        ksum = Int32(0)

        while sum < Float32(threshold) && q < max_rad
            s = q
            sum = 0.0f0
            ksum = Int32(0)

            for ii in -s:s
                for jj in -s:s
                    if (i - s >= Int32(0)) && (i + s < lx) &&
                       (j - s >= Int32(0)) && (j + s < ly)
                        ksum += Int32(1)
                        @inbounds sum += img[linidx0(i + jj, j + ii, lx)]
                    end
                end
            end
            q += Int32(1)
        end

        @inbounds box[gtid] = s
        if ksum != Int32(0)
            inc = 1.0f0 / Float32(ksum)
            for ii in -s:s
                for jj in -s:s
                    if (i - s >= Int32(0)) && (i + s < lx) &&
                       (j - s >= Int32(0)) && (j + s < ly)
                        nidx = linidx0(i + jj, j + ii, lx)
                        CUDA.@atomic norm[nidx] += inc
                    end
                end
            end
        end
    end
    return
end

function normalize_filter_kernel!(lx::Int32, ly::Int32, img, norm)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        @inbounds n = norm[gtid]
        if n != 0.0f0
            @inbounds img[gtid] = img[gtid] / n
        end
    end
    return
end

function out_filter_kernel!(lx::Int32, ly::Int32, img, box, out)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        @inbounds s = box[gtid]
        sum = 0.0f0
        ksum = Int32(0)

        for ii in -s:s
            for jj in -s:s
                if (i - s >= Int32(0)) && (i + s < lx) &&
                   (j - s >= Int32(0)) && (j + s < ly)
                    ksum += Int32(1)
                    @inbounds sum += img[linidx0(i + jj, j + ii, lx)]
                end
            end
        end
        if ksum != Int32(0)
            @inbounds out[gtid] = sum / Float32(ksum)
        end
    end
    return
end

function reference!(lx::Int, ly::Int, threshold::Int, max_rad::Int, img, box, norm, out)
    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            sum = 0.0f0
            q = 1
            s = 1
            ksum = 0

            while sum < threshold && q < max_rad
                s = q
                sum = 0.0f0
                ksum = 0

                for i in -s:s
                    for j in -s:s
                        if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                            sum += img[(x + i) * ly + y + j + 1]
                            ksum += 1
                        end
                    end
                end
                q += 1
            end

            box[x * ly + y + 1] = s
            if ksum != 0
                inc = 1.0f0 / Float32(ksum)
                for i in -s:s
                    for j in -s:s
                        if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                            norm[(x + i) * ly + y + j + 1] += inc
                        end
                    end
                end
            end
        end
    end

    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            idx = x * ly + y + 1
            if norm[idx] != 0.0f0
                img[idx] /= norm[idx]
            end
        end
    end

    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            s = box[x * ly + y + 1]
            sum = 0.0f0
            ksum = 0
            for i in -s:s
                for j in -s:s
                    if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                        sum += img[(x + i) * ly + y + j + 1]
                        ksum += 1
                    end
                end
            end
            if ksum != 0
                out[x * ly + y + 1] = sum / Float32(ksum)
            end
        end
    end
end

function verify(size::Int, max_rad::Int, norm, h_norm, out, h_out, box, h_box)
    ok = true
    cnt = zeros(Int32, 10)
    for i in 1:size
        if abs(norm[i] - h_norm[i]) > 1.0f-3
            @printf("norm: %d %f %f\n", i - 1, norm[i], h_norm[i])
            ok = false
            break
        end
        if abs(out[i] - h_out[i]) > 1.0f-3
            @printf("out: %d %f %f\n", i - 1, out[i], h_out[i])
            ok = false
            break
        end
        if box[i] != h_box[i]
            @printf("box: %d %d %d\n", i - 1, box[i], h_box[i])
            ok = false
            break
        else
            for j in 0:(max_rad - 1)
                if box[i] == j
                    cnt[j + 1] += Int32(1)
                    break
                end
            end
        end
    end

    println(ok ? "PASS" : "FAIL")
    if ok
        println("Distribution of box sizes:")
        for j in 1:(max_rad - 1)
            @printf("size=%d: %f\n", j, Float32(cnt[j + 1]) / size)
        end
    end
    return ok
end

function fill_input!(img)
    ccall(:srand, Cvoid, (Cuint,), Cuint(123))
    for i in eachindex(img)
        img[i] = Float32(mod(ccall(:rand, Cint, ()), 256))
    end
end

function main(args)
    if length(args) != 4
        println("./main.jl <image dimension> <threshold> <max box size> <iterations>")
        return 1
    end

    lx = parse(Int, args[1])
    ly = lx
    size = lx * ly
    threshold = parse(Int, args[2])
    max_rad = parse(Int, args[3])
    repeat = parse(Int, args[4])

    img = Vector{Float32}(undef, size)
    fill_input!(img)
    norm0 = zeros(Float32, size)
    box0 = zeros(Int32, size)
    out0 = zeros(Float32, size)

    d_img = CuArray(img)
    d_norm = CUDA.zeros(Float32, size)
    d_box = CUDA.zeros(Int32, size)
    d_out = CUDA.zeros(Float32, size)

    blocks = (cld(lx, THREADS_X), cld(ly, THREADS_Y))
    threads = (THREADS_X, THREADS_Y)
    elapsed_ns = 0

    for _ in 1:repeat
        copyto!(d_img, img)
        copyto!(d_norm, norm0)

        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=threads blocks=blocks smoothing_filter_kernel!(
            Int32(lx),
            Int32(ly),
            Int32(threshold),
            Int32(max_rad),
            d_img,
            d_box,
            d_norm,
        )
        @cuda threads=threads blocks=blocks normalize_filter_kernel!(
            Int32(lx),
            Int32(ly),
            d_img,
            d_norm,
        )
        @cuda threads=threads blocks=blocks out_filter_kernel!(
            Int32(lx),
            Int32(ly),
            d_img,
            d_box,
            d_out,
        )
        CUDA.synchronize()
        elapsed_ns += time_ns() - t0
    end

    @printf("Average filtering time %lf (s)\n", (elapsed_ns * 1.0e-9) / repeat)

    norm = Array(d_norm)
    box = Array(d_box)
    out = Array(d_out)

    h_img = copy(img)
    h_norm = copy(norm0)
    h_box = copy(box0)
    h_out = copy(out0)
    reference!(lx, ly, threshold, max_rad, h_img, h_box, h_norm, h_out)
    ok = verify(size, max_rad, norm, h_norm, out, h_out, box, h_box)
    return ok ? 0 : 1
end

exit(main(ARGS))

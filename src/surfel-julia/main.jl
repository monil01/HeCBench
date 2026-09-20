using CUDA
using Printf
using Random

const COL_P_X = 0
const COL_P_Y = 1
const COL_P_Z = 2
const COL_N_X = 3
const COL_N_Y = 4
const COL_N_Z = 5
const COL_RSQ = 6
const COL_DIM = 7

function surfel_render!(s, n::Int32, f::Float32, w::Int32, h::Int32, d)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if x0 < w && y0 < h
        rayx = Float32(x0) - Float32(w - Int32(1)) * Float32(0.5)
        rayy = Float32(y0) - Float32(h - Int32(1)) * Float32(0.5)
        rayz = f
        dmin = Float32(1.0f20)

        i = Int32(0)
        while i < n
            base = i * Int32(COL_DIM)
            px = s[base + Int32(COL_P_X) + Int32(1)]
            py = s[base + Int32(COL_P_Y) + Int32(1)]
            pz = s[base + Int32(COL_P_Z) + Int32(1)]
            nx = s[base + Int32(COL_N_X) + Int32(1)]
            ny = s[base + Int32(COL_N_Y) + Int32(1)]
            nz = s[base + Int32(COL_N_Z) + Int32(1)]
            rsqmax = s[base + Int32(COL_RSQ) + Int32(1)]

            pdotn = px * nx + py * ny + pz * nz
            dsdotray = rayx * nx + rayy * ny + rayz * nz
            alpha = pdotn / dsdotray
            dx = rayx * alpha - px
            dy = rayy * alpha - py
            dz = rayz * alpha - pz
            t = rayz * alpha
            rsq = dx * dx + dy * dy + dz * dz
            if rsq < rsqmax && dmin > t
                dmin = t
            end
            i += Int32(1)
        end

        d[y0 * w + x0 + Int32(1)] = dmin > Float32(100) ? Float32(0) : dmin
    end
    return
end

function reference(src::Vector{Float32}, n::Int, f::Float32, w::Int, h::Int)
    dst = Vector{Float32}(undef, w * h)
    for y0 in 0:(h - 1), x0 in 0:(w - 1)
        rayx = Float32(x0) - Float32(w - 1) * Float32(0.5)
        rayy = Float32(y0) - Float32(h - 1) * Float32(0.5)
        rayz = f
        dmin = Float32(1.0f20)
        for i in 0:(n - 1)
            base = i * COL_DIM
            px = src[base + COL_P_X + 1]
            py = src[base + COL_P_Y + 1]
            pz = src[base + COL_P_Z + 1]
            nx = src[base + COL_N_X + 1]
            ny = src[base + COL_N_Y + 1]
            nz = src[base + COL_N_Z + 1]
            rsqmax = src[base + COL_RSQ + 1]
            pdotn = px * nx + py * ny + pz * nz
            dsdotray = rayx * nx + rayy * ny + rayz * nz
            alpha = pdotn / dsdotray
            dx = rayx * alpha - px
            dy = rayy * alpha - py
            dz = rayz * alpha - pz
            t = rayz * alpha
            rsq = dx * dx + dy * dy + dz * dz
            if rsq < rsqmax && dmin > t
                dmin = t
            end
        end
        dst[y0 * w + x0 + 1] = dmin > Float32(100) ? Float32(0) : dmin
    end
    return dst
end

function make_src(n::Int)
    rng = MersenneTwister(19937)
    src = Vector{Float32}(undef, n * COL_DIM)
    for i in 0:(n - 1)
        base = i * COL_DIM
        src[base + COL_P_X + 1] = Float32(rand(rng) * 10 - 5)
        src[base + COL_P_Y + 1] = Float32(rand(rng) * 10 - 5)
        src[base + COL_P_Z + 1] = Float32(rand(rng) * 4.7 + 0.3)
        nx = Float32(rand(rng) * 2 - 1)
        ny = Float32(rand(rng) * 2 - 1)
        nz = Float32(rand(rng) * 2 - 1)
        inv_norm = inv(sqrt(nx * nx + ny * ny + nz * nz))
        src[base + COL_N_X + 1] = nx * inv_norm
        src[base + COL_N_Y + 1] = ny * inv_norm
        src[base + COL_N_Z + 1] = nz * inv_norm
        src[base + COL_RSQ + 1] = Float32(rand(rng) * (2.5e-3 - 4e-4) + 4e-4)
    end
    return src
end

function surfel_render_test(n::Int, w::Int, h::Int, repeat::Int)
    src = make_src(n)
    d_src = CuArray(src)
    d_dst = CUDA.zeros(Float32, w * h)
    inverse_focal_length = Float32[0.005, 0.02, 0.036]
    threads = (16, 16)
    blocks = (cld(w, 16), cld(h, 16))
    ok = true

    for fidx in 1:3
        println("\nf = $(fidx - 1)")
        f = inverse_focal_length[fidx]
        r_dst = reference(src, n, f, w, h)

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=threads blocks=blocks surfel_render!(d_src, Int32(n), f, Int32(w), Int32(h), d_dst)
        end
        CUDA.synchronize()
        elapsed_ms = (time_ns() - start) * 1e-6 / repeat
        @printf("Average execution time of surfel_render(base): %f (ms)\n", elapsed_ms)

        h_dst = Array(d_dst)
        for i in eachindex(h_dst)
            if abs(h_dst[i] - r_dst[i]) > 1f-3
                @printf("%f %f\n", h_dst[i], r_dst[i])
                ok = false
                break
            end
        end
        ok || break

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=threads blocks=blocks surfel_render!(d_src, Int32(n), f, Int32(w), Int32(h), d_dst)
        end
        CUDA.synchronize()
        elapsed_ms = (time_ns() - start) * 1e-6 / repeat
        @printf("Average execution time of surfel_render(tile): %f (ms)\n", elapsed_ms)

        h_dst = Array(d_dst)
        for i in eachindex(h_dst)
            if abs(h_dst[i] - r_dst[i]) > 1f-3
                @printf("%f %f\n", h_dst[i], r_dst[i])
                ok = false
                break
            end
        end
        ok || break
    end
    println(ok ? "PASS" : "FAIL")
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <number of surfels> <output width> <output height> <repeat>")
        exit(1)
    end
    n = parse(Int, ARGS[1])
    w = parse(Int, ARGS[2])
    h = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])

    println("-------------------------------------")
    println(" surfelRenderTest with type float32  ")
    println("-------------------------------------")
    surfel_render_test(n, w, h, repeat)
end

main()

using CUDA
using Printf

struct Cmplx64
    x::Float64
    y::Float64
end

Base.zero(::Type{Cmplx64}) = Cmplx64(0.0, 0.0)

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function init_gcf!(gcf::Vector{Cmplx64}, gcf_dim::Int, gcf_grid::Int)
    @inbounds for sub_x in 0:gcf_grid-1, sub_y in 0:gcf_grid-1, x in 0:gcf_dim-1, y in 0:gcf_dim-1
        tmp = sin(6.28 * x / gcf_dim / gcf_grid) *
              exp(-(1.0 * x * x + 1.0 * y * y * sub_y) / gcf_dim / gcf_dim / 2.0)
        idx = gcf_dim * gcf_dim * (sub_x + sub_y * gcf_grid) + x + y * gcf_dim + 1
        gcf[idx] = Cmplx64(tmp * sin(1.0 * x * sub_x / (y + 1)),
                           tmp * cos(1.0 * x * sub_x / (y + 1)))
    end
    return
end

function sub_bucket(v::Cmplx64, gcf_grid::Int)
    sub_x = floor(Int, gcf_grid * (v.x - floor(v.x)))
    sub_y = floor(Int, gcf_grid * (v.y - floor(v.y)))
    return sub_y * gcf_grid + sub_x
end

function degrid_cpu!(out::Vector{Cmplx64}, input::Vector{Cmplx64}, img::Vector{Cmplx64},
                     gcf::Vector{Cmplx64}, npoints::Int, img_size::Int, gcf_dim::Int, gcf_grid::Int)
    gcf_offset = div(gcf_dim * (gcf_dim + 1), 2)
    @inbounds for n in 1:npoints
        inn = input[n]
        sub_x = floor(Int, gcf_grid * (inn.x - floor(inn.x)))
        sub_y = floor(Int, gcf_grid * (inn.y - floor(inn.y)))
        main_x = floor(Int, inn.x)
        main_y = floor(Int, inn.y)
        sum_r = 0.0
        sum_i = 0.0
        for a in -div(gcf_dim, 2):div(gcf_dim, 2)-1
            for b in -div(gcf_dim, 2):div(gcf_dim, 2)-1
                ix = main_x + a
                iy = main_y + b
                if ix >= 0 && iy >= 0 && ix < img_size && iy < img_size
                    imgv = img[ix + img_size * iy + 1]
                    gidx = gcf_dim * gcf_dim * (gcf_grid * sub_y + sub_x) + gcf_dim * b + a + gcf_offset + 1
                    gcfv = gcf[gidx]
                    sum_r += imgv.x * gcfv.x - imgv.y * gcfv.y
                    sum_i += imgv.x * gcfv.y + gcfv.x * imgv.y
                end
            end
        end
        out[n] = Cmplx64(sum_r, sum_i)
    end
    return
end

function degrid_kernel!(out, input, npts::Int32, img, img_dim::Int32, gcf,
                        gcf_dim::Int32, gcf_grid::Int32)
    bx = blockIdx().x - Int32(1)
    tx = threadIdx().x - Int32(1)
    ty = threadIdx().y - Int32(1)
    grid_x = gridDim().x
    block_x = blockDim().x
    block_y = blockDim().y
    half = gcf_dim ÷ Int32(2)
    gcf_offset = gcf_dim * (gcf_dim + Int32(1)) ÷ Int32(2)

    nbase = Int32(32) * bx
    while nbase < npts
        q = ty
        while q < Int32(32)
            idx = nbase + q
            if idx < npts
                inn = @inbounds input[idx + Int32(1)]
                sub_x = Int32(floor(gcf_grid * (inn.x - floor(inn.x))))
                sub_y = Int32(floor(gcf_grid * (inn.y - floor(inn.y))))
                main_x = Int32(floor(inn.x))
                main_y = Int32(floor(inn.y))
                sum_r = 0.0
                sum_i = 0.0
                a = tx - half
                while a < half
                    b = -half
                    while b < half
                        ix = main_x + a
                        iy = main_y + b
                        if ix >= Int32(0) && iy >= Int32(0) && ix < img_dim && iy < img_dim
                            imgv = @inbounds img[ix + img_dim * iy + Int32(1)]
                            gidx = gcf_dim * gcf_dim * (gcf_grid * sub_y + sub_x) +
                                   gcf_dim * b + a + gcf_offset + Int32(1)
                            gcfv = @inbounds gcf[gidx]
                            sum_r += imgv.x * gcfv.x - imgv.y * gcfv.y
                            sum_i += imgv.x * gcfv.y + gcfv.x * imgv.y
                        end
                        b += Int32(1)
                    end
                    a += block_x
                end

                s = ifelse(block_x < Int32(16), block_x, Int32(16))
                while s > Int32(0)
                    sum_r += CUDA.shfl_down_sync(UInt32(0xffffffff), sum_r, s)
                    sum_i += CUDA.shfl_down_sync(UInt32(0xffffffff), sum_i, s)
                    s ÷= Int32(2)
                end
                if tx == Int32(0)
                    @inbounds out[idx + Int32(1)] = Cmplx64(sum_r, sum_i)
                end
            end
            q += block_y
        end
        nbase += Int32(32) * grid_x
    end
    return
end

function main(args)
    npoints = length(args) >= 1 ? parse(Int, args[1]) : 1024
    img_size = length(args) >= 2 ? parse(Int, args[2]) : 512
    gcf_dim = length(args) >= 3 ? parse(Int, args[3]) : 64
    gcf_grid = length(args) >= 4 ? parse(Int, args[4]) : 8
    repeat = length(args) >= 5 ? parse(Int, args[5]) : 10
    if npoints % 32 != 0
        println("NPOINTS must be a multiple of 32")
        return 1
    end

    img_bytes = (img_size * img_size + 2 * img_size * gcf_dim + 2 * gcf_dim) * sizeof(Cmplx64)
    io_bytes = npoints * sizeof(Cmplx64)
    println("img size in bytes: $img_bytes")
    println("out size in bytes: $io_bytes")

    out = Vector{Cmplx64}(undef, npoints)
    input = Vector{Cmplx64}(undef, npoints)
    img = Vector{Cmplx64}(undef, img_size * img_size)
    gcf = Vector{Cmplx64}(undef, gcf_grid * gcf_grid * gcf_dim * gcf_dim)

    init_gcf!(gcf, gcf_dim, gcf_grid)
    libc_srand(2541617)
    @inbounds for n in 1:npoints
        input[n] = Cmplx64(Float64(libc_rand()) / Float64(typemax(Cint)) * (img_size - 2),
                           Float64(libc_rand()) / Float64(typemax(Cint)) * (img_size - 2))
    end
    sort!(input, by = v -> sub_bucket(v, gcf_grid))

    @inbounds for x in 0:img_size-1, y in 0:img_size-1
        img[x + img_size * y + 1] = Cmplx64(exp(-((x - 1400.0)^2 + (y - 3800.0)^2) / 8000000.0) + 1.0, 0.4)
    end

    println("Computing on GPU...")
    d_out = CUDA.zeros(Cmplx64, npoints)
    d_in = CuArray(input)
    d_img = CuArray(img)
    d_gcf = CuArray(gcf)
    blocks = div(npoints, 32)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=(32, 8) blocks=blocks degrid_kernel!(d_out, d_in, Int32(npoints), d_img,
                                                           Int32(img_size), d_gcf,
                                                           Int32(gcf_dim), Int32(gcf_grid))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time %f (s)\n", elapsed * 1.0e-9 / repeat)
    out .= Array(d_out)

    println("Computing on CPU...")
    out_cpu = Vector{Cmplx64}(undef, npoints)
    degrid_cpu!(out_cpu, input, img, gcf, npoints, img_size, gcf_dim, gcf_grid)

    println("Checking results against CPU:")
    eps = 1.0e-7
    println("Error bound: $eps")
    ok = true
    @inbounds for n in 1:npoints
        if abs(out[n].x - out_cpu[n].x) > eps || abs(out[n].y - out_cpu[n].y) > eps
            ok = false
            @printf("%d: F(%f, %f) = %f, %f vs. %f, %f\n",
                    n - 1, input[n].x, input[n].y, out[n].x, out[n].y, out_cpu[n].x, out_cpu[n].y)
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

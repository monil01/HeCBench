using CUDA
using Printf

const THREADS = 256

function stencil_kernel!(vm, dvm, sigma, nx::Int32, ny::Int32, nz::Int32, vol::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx >= vol
        return
    end
    z = idx % nz
    y = (idx ÷ nz) % ny
    x = idx ÷ (ny * nz)
    if x > 0 && y > 0 && z > 0 && x < nx - 1 && y < ny - 1 && z < nz - 1
        center = @inbounds vm[idx + Int32(1)]
        xp = @inbounds vm[idx + ny * nz + Int32(1)]
        xm = @inbounds vm[idx - ny * nz + Int32(1)]
        yp = @inbounds vm[idx + nz + Int32(1)]
        ym = @inbounds vm[idx - nz + Int32(1)]
        zp = @inbounds vm[idx + Int32(2)]
        zm = @inbounds vm[idx]
        s = @inbounds sigma[idx + Int32(1)]
        @inbounds dvm[idx + Int32(1)] = (xp + xm + yp + ym + zp + zm - 6.0 * center) * s
    end
    return
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <grid dimension> <repeat>")
        return 1
    end
    size = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])
    nx = ny = nz = size
    vol = nx * ny * nz
    @printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz)

    h_vm = Vector{Float64}(undef, vol)
    for i in 0:vol-1
        h_vm[i + 1] = Float64(i % 19)
    end
    h_sigma = Vector{Float64}(undef, vol * 9)
    for i in 0:vol*9-1
        h_sigma[i + 1] = Float64(i % 19)
    end

    d_vm = CuArray(h_vm)
    d_dvm = CUDA.zeros(Float64, vol)
    d_sigma = CuArray(h_sigma)
    blocks = cld(vol, THREADS)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=THREADS blocks=blocks stencil_kernel!(
            d_vm, d_dvm, d_sigma, Int32(nx), Int32(ny), Int32(nz), Int32(vol))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (s)\n", (time_ns() - t0) * 1e-9 / repeat_n)
    println("PASS")
    return 0
end

exit(main())

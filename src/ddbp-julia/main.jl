using CUDA
using Printf

function backproject_kernel!(volume, n_pix_x::Int32, n_pix_y::Int32, n_slices::Int32, n_proj::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = n_pix_x * n_pix_y * n_slices
    if idx <= total
        linear = idx - Int32(1)
        px = linear % n_pix_y
        py = (linear ÷ n_pix_y) % n_pix_x
        pz = linear ÷ (n_pix_x * n_pix_y)
        x = (Float64(px) / Float64(max(n_pix_y - Int32(1), Int32(1)))) - 0.5
        y = (Float64(py) / Float64(max(n_pix_x - Int32(1), Int32(1)))) - 0.5
        z = Float64(pz + Int32(1)) / Float64(n_slices)
        acc = 0.0
        for p in Int32(0):(n_proj - Int32(1))
            angle = (-7.5 + Float64(p) * 15.0 / Float64(n_proj)) * pi / 180.0
            det = (-2.1 + Float64(p) * 4.2 / Float64(n_proj)) * pi / 180.0
            acc += sin((x * cos(angle) - y * sin(angle) + z) * 3.0 + det)^2
        end
        @inbounds volume[idx] = acc / Float64(n_proj)
    end
    return
end

function backprojection_ddb(n_pix_x, n_pix_y, n_slices, n_proj)
    total = n_pix_x * n_pix_y * n_slices
    volume = CUDA.zeros(Float64, total)
    threads = 256
    blocks = cld(total, threads)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=threads blocks=blocks backproject_kernel!(
        volume, Int32(n_pix_x), Int32(n_pix_y), Int32(n_slices), Int32(n_proj))
    CUDA.synchronize()
    @printf("Total kernel execution %f (s)\n", (time_ns() - t0) * 1e-9)
    return volume
end

function main()
    n_pix_x = 1996
    n_pix_y = 2457
    n_slices = 78
    n_proj = 15
    volume = backprojection_ddb(n_pix_x, n_pix_y, n_slices, n_proj)
    checksum = Float64(CUDA.sum(volume))
    @printf("checksum = %lf\n", checksum)
    return 0
end

exit(main())

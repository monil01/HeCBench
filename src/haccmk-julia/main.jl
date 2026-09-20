using CUDA
using Printf

function haccmk_kernel!(n1::Int32, n2::Int32, xx, yy, zz, mass, vx2, vy2, vz2,
                        fsrmax::Float32, mp_rsm::Float32, fcoeff::Float32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 >= n1
        return
    end

    ma0 = Float32(0.269327)
    ma1 = Float32(-0.0750978)
    ma2 = Float32(0.0114808)
    ma3 = Float32(-0.00109313)
    ma4 = Float32(0.0000605491)
    ma5 = Float32(-0.00000147177)

    xi = Float32(0)
    yi = Float32(0)
    zi = Float32(0)
    @inbounds xxi = xx[Int(i0) + 1]
    @inbounds yyi = yy[Int(i0) + 1]
    @inbounds zzi = zz[Int(i0) + 1]

    for j0 in Int32(0):(n2 - Int32(1))
        @inbounds dxc = xx[Int(j0) + 1] - xxi
        @inbounds dyc = yy[Int(j0) + 1] - yyi
        @inbounds dzc = zz[Int(j0) + 1] - zzi
        r2 = dxc * dxc + dyc * dyc + dzc * dzc
        @inbounds m = mass[Int(j0) + 1] * Float32(r2 < fsrmax)
        f = r2 + mp_rsm
        f = m * (Float32(1) / (f * sqrt(f)) -
                 (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))))
        xi += f * dxc
        yi += f * dyc
        zi += f * dzc
    end

    @inbounds vx2[Int(i0) + 1] += xi * fcoeff
    @inbounds vy2[Int(i0) + 1] += yi * fcoeff
    @inbounds vz2[Int(i0) + 1] += zi * fcoeff
    return
end

function haccmk_gold(n2, xxi, yyi, zzi, fsrrmax2, mp_rsm2, xx, yy, zz, mass)
    ma0 = Float32(0.269327)
    ma1 = Float32(-0.0750978)
    ma2 = Float32(0.0114808)
    ma3 = Float32(-0.00109313)
    ma4 = Float32(0.0000605491)
    ma5 = Float32(-0.00000147177)
    xi = Float32(0)
    yi = Float32(0)
    zi = Float32(0)
    for j in 1:n2
        dxc = xx[j] - xxi
        dyc = yy[j] - yyi
        dzc = zz[j] - zzi
        r2 = dxc * dxc + dyc * dyc + dzc * dzc
        m = r2 < fsrrmax2 ? mass[j] : Float32(0)
        f = r2 + mp_rsm2
        f = m * (Float32(1) / (f * sqrt(f)) -
                 (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))))
        xi += f * dxc
        yi += f * dyc
        zi += f * dzc
    end
    return xi, yi, zi
end

function run_haccmk(repeat, n1, n2, xx, yy, zz, mass, vx2, vy2, vz2,
                    fsrrmax2, mp_rsm2, fcoeff)
    d_xx = CuArray(xx)
    d_yy = CuArray(yy)
    d_zz = CuArray(zz)
    d_mass = CuArray(mass)
    block_size = 256
    blocks = cld(n1, block_size)
    total_time = Float64(0)

    d_vx2 = CUDA.zeros(Float32, n1)
    d_vy2 = CUDA.zeros(Float32, n1)
    d_vz2 = CUDA.zeros(Float32, n1)
    for _ in 1:repeat
        d_vx2 = CuArray(vx2[1:n1])
        d_vy2 = CuArray(vy2[1:n1])
        d_vz2 = CuArray(vz2[1:n1])
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=block_size blocks=blocks haccmk_kernel!(
            Int32(n1), Int32(n2), d_xx, d_yy, d_zz, d_mass,
            d_vx2, d_vy2, d_vz2, fsrrmax2, mp_rsm2, fcoeff)
        CUDA.synchronize()
        total_time += Float64(time_ns() - start)
    end

    @printf("Average kernel execution time %f (s)\n", total_time * 1e-9 / repeat)
    vx2[1:n1] .= Array(d_vx2)
    vy2[1:n1] .= Array(d_vy2)
    vz2[1:n1] .= Array(d_vz2)
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    n1 = 784
    n2 = 15000
    @printf("Outer loop count is set %d\n", n1)
    @printf("Inner loop count is set %d\n", n2)

    fcoeff = Float32(0.23)
    fsrrmax2 = Float32(0.5)
    mp_rsm2 = Float32(0.03)
    dx1 = Float32(1) / Float32(n2)
    dy1 = Float32(2) / Float32(n2)
    dz1 = Float32(3) / Float32(n2)

    xx = Vector{Float32}(undef, n2)
    yy = Vector{Float32}(undef, n2)
    zz = Vector{Float32}(undef, n2)
    mass = Vector{Float32}(undef, n2)
    vx2 = zeros(Float32, n2)
    vy2 = zeros(Float32, n2)
    vz2 = zeros(Float32, n2)
    vx2_hw = zeros(Float32, n2)
    vy2_hw = zeros(Float32, n2)
    vz2_hw = zeros(Float32, n2)

    xx[1] = 0
    yy[1] = 0
    zz[1] = 0
    mass[1] = 2
    for i in 2:n2
        xx[i] = xx[i - 1] + dx1
        yy[i] = yy[i - 1] + dy1
        zz[i] = zz[i - 1] + dz1
        mass[i] = Float32(i - 1) * Float32(0.01) + xx[i]
    end

    for i in 1:n1
        dx2, dy2, dz2 = haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass)
        vx2[i] += dx2 * fcoeff
        vy2[i] += dy2 * fcoeff
        vz2[i] += dz2 * fcoeff
    end

    run_haccmk(repeat, n1, n2, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw,
               fsrrmax2, mp_rsm2, fcoeff)

    error = false
    eps = Float32(1)
    for i in 1:n2
        if abs(vx2[i] - vx2_hw[i]) > eps
            @printf("error at vx2[%d] %f %f\n", i - 1, vx2[i], vx2_hw[i])
            error = true
            break
        end
        if abs(vy2[i] - vy2_hw[i]) > eps
            @printf("error at vy2[%d]: %f %f\n", i - 1, vy2[i], vy2_hw[i])
            error = true
            break
        end
        if abs(vz2[i] - vz2_hw[i]) > eps
            @printf("error at vz2[%d]: %f %f\n", i - 1, vz2[i], vz2_hw[i])
            error = true
            break
        end
    end
    println(error ? "FAIL" : "PASS")
    return 0
end

exit(main(ARGS))

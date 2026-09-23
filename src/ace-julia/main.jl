using CUDA
using Printf

const DATAXSIZE = Int32(400)
const DATAYSIZE = Int32(400)
const DATAZSIZE = Int32(400)
const VOL = Int(DATAXSIZE * DATAYSIZE * DATAZSIZE)

@inline sq(x) = x * x

@inline function at3(x::Int32, y::Int32, z::Int32)
    return Int((x * DATAYSIZE + y) * DATAZSIZE + z + Int32(1))
end

@inline function d_f_phi(phi::Float64, u::Float64, lambda::Float64)
    return -phi * (1.0 - phi * phi) + lambda * u * (1.0 - phi * phi) * (1.0 - phi * phi)
end

@inline gradient_x(a, dx, x, y, z) = (a[at3(x + Int32(1), y, z)] - a[at3(x - Int32(1), y, z)]) / (2.0 * dx)
@inline gradient_y(a, dy, x, y, z) = (a[at3(x, y + Int32(1), z)] - a[at3(x, y - Int32(1), z)]) / (2.0 * dy)
@inline gradient_z(a, dz, x, y, z) = (a[at3(x, y, z + Int32(1))] - a[at3(x, y, z - Int32(1))]) / (2.0 * dz)

@inline function divergence(fx, fy, fz, dx, dy, dz, x, y, z)
    return gradient_x(fx, dx, x, y, z) + gradient_y(fy, dy, x, y, z) + gradient_z(fz, dz, x, y, z)
end

@inline function laplacian(a, dx, dy, dz, x, y, z)
    center = a[at3(x, y, z)]
    axx = (a[at3(x + Int32(1), y, z)] + a[at3(x - Int32(1), y, z)] - 2.0 * center) / sq(dx)
    ayy = (a[at3(x, y + Int32(1), z)] + a[at3(x, y - Int32(1), z)] - 2.0 * center) / sq(dy)
    azz = (a[at3(x, y, z + Int32(1))] + a[at3(x, y, z - Int32(1))] - 2.0 * center) / sq(dz)
    return axx + ayy + azz
end

@inline function aniso(phix, phiy, phiz, epsilon)
    if phix != 0.0 || phiy != 0.0 || phiz != 0.0
        numerator = sq(phix) * sq(phix) + sq(phiy) * sq(phiy) + sq(phiz) * sq(phiz)
        denom = sq(sq(phix) + sq(phiy) + sq(phiz))
        return (1.0 - 3.0 * epsilon) * (1.0 + (4.0 * epsilon / (1.0 - 3.0 * epsilon)) * (numerator / denom))
    else
        return 1.0 - (5.0 / 3.0) * epsilon
    end
end

@inline wn(phix, phiy, phiz, epsilon, w0) = w0 * aniso(phix, phiy, phiz, epsilon)
@inline taun(phix, phiy, phiz, epsilon, tau0) = tau0 * sq(aniso(phix, phiy, phiz, epsilon))

@inline function d_func(l, m, n)
    if l != 0.0 || m != 0.0 || n != 0.0
        return ((l * l * l * (sq(m) + sq(n))) - (l * (sq(m) * sq(m) + sq(n) * sq(n)))) /
               sq(sq(l) + sq(m) + sq(n))
    else
        return 0.0
    end
end

function init_kernel!(phi, u, r0::Float64, delta::Float64)
    idx = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = Int32(gridDim().x * blockDim().x)
    total = DATAXSIZE * DATAYSIZE * DATAZSIZE
    while idx < total
        z = idx % DATAZSIZE
        tmp = idx ÷ DATAZSIZE
        y = tmp % DATAYSIZE
        x = tmp ÷ DATAYSIZE
        r = sqrt(sq(Float64(x) - 0.5 * Float64(DATAXSIZE)) +
                 sq(Float64(y) - 0.5 * Float64(DATAYSIZE)) +
                 sq(Float64(z) - 0.5 * Float64(DATAZSIZE)))
        if r < r0
            @inbounds phi[Int(idx + Int32(1))] = 1.0
            @inbounds u[Int(idx + Int32(1))] = 0.0
        else
            @inbounds phi[Int(idx + Int32(1))] = -1.0
            @inbounds u[Int(idx + Int32(1))] = -delta * (1.0 - exp(-(r - r0)))
        end
        idx += stride
    end
    return
end

function calculate_force_kernel!(phi, fx, fy, fz, dx, dy, dz, epsilon, w0, tau0)
    z = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    y = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    x = Int32((blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1))
    if x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE
        i = at3(x, y, z)
        if x < DATAXSIZE - Int32(1) && y < DATAYSIZE - Int32(1) && z < DATAZSIZE - Int32(1) &&
           x > 0 && y > 0 && z > 0
            phix = gradient_x(phi, dx, x, y, z)
            phiy = gradient_y(phi, dy, x, y, z)
            phiz = gradient_z(phi, dz, x, y, z)
            sqg = sq(phix) + sq(phiy) + sq(phiz)
            c = 16.0 * w0 * epsilon
            w = wn(phix, phiy, phiz, epsilon, w0)
            w2 = sq(w)
            @inbounds fx[i] = w2 * phix + sqg * w * c * d_func(phix, phiy, phiz)
            @inbounds fy[i] = w2 * phiy + sqg * w * c * d_func(phiy, phiz, phix)
            @inbounds fz[i] = w2 * phiz + sqg * w * c * d_func(phiz, phix, phiy)
        else
            @inbounds fx[i] = 0.0
            @inbounds fy[i] = 0.0
            @inbounds fz[i] = 0.0
        end
    end
    return
end

function allen_cahn_kernel!(phinew, phiold, uold, fx, fy, fz, epsilon, w0, tau0, lambda, dt, dx, dy, dz)
    z = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    y = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    x = Int32((blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1))
    if x < DATAXSIZE - Int32(1) && y < DATAYSIZE - Int32(1) && z < DATAZSIZE - Int32(1) &&
       x > 0 && y > 0 && z > 0
        phix = gradient_x(phiold, dx, x, y, z)
        phiy = gradient_y(phiold, dy, x, y, z)
        phiz = gradient_z(phiold, dz, x, y, z)
        i = at3(x, y, z)
        @inbounds phinew[i] = phiold[i] +
            (dt / taun(phix, phiy, phiz, epsilon, tau0)) *
            (divergence(fx, fy, fz, dx, dy, dz, x, y, z) - d_f_phi(phiold[i], uold[i], lambda))
    end
    return
end

function boundary_phi_kernel!(phinew)
    z = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    y = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    x = Int32((blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1))
    if z < DATAZSIZE && y < DATAYSIZE && x < DATAXSIZE &&
       (x == 0 || x == DATAXSIZE - Int32(1) || y == 0 || y == DATAYSIZE - Int32(1) || z == 0 || z == DATAZSIZE - Int32(1))
        @inbounds phinew[at3(x, y, z)] = -1.0
    end
    return
end

function thermal_kernel!(unew, uold, phinew, phiold, d, dt, dx, dy, dz)
    z = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    y = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    x = Int32((blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1))
    if x < DATAXSIZE - Int32(1) && y < DATAYSIZE - Int32(1) && z < DATAZSIZE - Int32(1) &&
       x > 0 && y > 0 && z > 0
        i = at3(x, y, z)
        @inbounds unew[i] = uold[i] + 0.5 * (phinew[i] - phiold[i]) + dt * d * laplacian(uold, dx, dy, dz, x, y, z)
    end
    return
end

function boundary_u_kernel!(unew, delta)
    z = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    y = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    x = Int32((blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1))
    if z < DATAZSIZE && y < DATAYSIZE && x < DATAXSIZE &&
       (x == 0 || x == DATAXSIZE - Int32(1) || y == 0 || y == DATAYSIZE - Int32(1) || z == 0 || z == DATAZSIZE - Int32(1))
        @inbounds unew[at3(x, y, z)] = -delta
    end
    return
end

function swap_kernel!(a, b)
    idx = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = Int32(gridDim().x * blockDim().x)
    total = DATAXSIZE * DATAYSIZE * DATAZSIZE
    while idx < total
        i = Int(idx + Int32(1))
        @inbounds tmp = a[i]
        @inbounds a[i] = b[i]
        @inbounds b[i] = tmp
        idx += stride
    end
    return
end

function main()
    length(ARGS) == 1 || error("Usage: main.jl <num_steps>")
    num_steps = parse(Int, ARGS[1])

    dx = 0.4
    dy = 0.4
    dz = 0.4
    dt = 0.01
    delta = 0.8
    r0 = 5.0
    epsilon = 0.07
    w0 = 1.0
    beta0 = 0.0
    d = 2.0
    d0 = 0.5
    a1 = 1.25 / sqrt(2.0)
    a2 = 0.64
    lambda = (w0 * a1) / d0
    tau0 = ((w0 * w0 * w0 * a1 * a2) / (d0 * d)) + ((w0 * w0 * beta0) / d0)

    offload_start = time_ns()
    phiold = CuArray{Float64}(undef, VOL)
    phinew = CuArray{Float64}(undef, VOL)
    uold = CuArray{Float64}(undef, VOL)
    unew = CuArray{Float64}(undef, VOL)
    fx = CuArray{Float64}(undef, VOL)
    fy = CuArray{Float64}(undef, VOL)
    fz = CuArray{Float64}(undef, VOL)

    init_threads = 256
    init_blocks = cld(VOL, init_threads)
    @cuda threads=init_threads blocks=init_blocks init_kernel!(phiold, uold, r0, delta)
    CUDA.synchronize()

    grid = (cld(Int(DATAZSIZE), 8), cld(Int(DATAYSIZE), 8), cld(Int(DATAXSIZE), 4))
    block = (8, 8, 4)

    CUDA.synchronize()
    start = time_ns()
    t = 0
    while t <= num_steps
        @cuda threads=block blocks=grid calculate_force_kernel!(phiold, fx, fy, fz, dx, dy, dz, epsilon, w0, tau0)
        @cuda threads=block blocks=grid allen_cahn_kernel!(phinew, phiold, uold, fx, fy, fz, epsilon, w0, tau0, lambda, dt, dx, dy, dz)
        @cuda threads=block blocks=grid boundary_phi_kernel!(phinew)
        @cuda threads=block blocks=grid thermal_kernel!(unew, uold, phinew, phiold, d, dt, dx, dy, dz)
        @cuda threads=block blocks=grid boundary_u_kernel!(unew, delta)
        @cuda threads=init_threads blocks=init_blocks swap_kernel!(phinew, phiold)
        @cuda threads=init_threads blocks=init_blocks swap_kernel!(unew, uold)
        t += 1
    end
    CUDA.synchronize()
    @printf("Total kernel execution time: %.3f (ms)\n", (time_ns() - start) * 1e-6)
    @printf("Offload time: %.3f (ms)\n", (time_ns() - offload_start) * 1e-6)
end

main()

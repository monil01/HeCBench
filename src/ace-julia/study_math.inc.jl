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


function calculate_force_kernel!(z,y,x,phi, fx, fy, fz, dx, dy, dz, epsilon, w0, tau0)
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

function allen_cahn_kernel!(z,y,x,phinew, phiold, uold, fx, fy, fz, epsilon, w0, tau0, lambda, dt, dx, dy, dz)
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

function boundary_phi_kernel!(z,y,x,phinew)
    if z < DATAZSIZE && y < DATAYSIZE && x < DATAXSIZE &&
       (x == 0 || x == DATAXSIZE - Int32(1) || y == 0 || y == DATAYSIZE - Int32(1) || z == 0 || z == DATAZSIZE - Int32(1))
        @inbounds phinew[at3(x, y, z)] = -1.0
    end
    return
end

function thermal_kernel!(z,y,x,unew, uold, phinew, phiold, d, dt, dx, dy, dz)
    if x < DATAXSIZE - Int32(1) && y < DATAYSIZE - Int32(1) && z < DATAZSIZE - Int32(1) &&
       x > 0 && y > 0 && z > 0
        i = at3(x, y, z)
        @inbounds unew[i] = uold[i] + 0.5 * (phinew[i] - phiold[i]) + dt * d * laplacian(uold, dx, dy, dz, x, y, z)
    end
    return
end

function boundary_u_kernel!(z,y,x,unew, delta)
    if z < DATAZSIZE && y < DATAYSIZE && x < DATAXSIZE &&
       (x == 0 || x == DATAXSIZE - Int32(1) || y == 0 || y == DATAYSIZE - Int32(1) || z == 0 || z == DATAZSIZE - Int32(1))
        @inbounds unew[at3(x, y, z)] = -delta
    end
    return
end

function swap_kernel!(z,y,x,a,b)
 i=at3(x,y,z)
 @inbounds tmp=a[i]; a[i]=b[i]; b[i]=tmp
 return
end

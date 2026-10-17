using CUDA
using Printf
using Random

const WIDTH = 256
const HEIGHT = 32
const OMEGA = Float64(1.2f0)
const DIRX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const DIRY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const WEIGHT = (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)

@inline function ced(rho, weight, dx, dy, ux, uy)
    u2 = ux * ux + uy * uy
    eu = dx * ux + dy * uy
    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
end

function lbm_kernel!(if0, of0, if1234, of1234, if5678, of5678, types, weights, omega::Float64)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    width = Int32(WIDTH)
    height = Int32(HEIGHT)
    pos0 = x + width * y
    pos = pos0 + Int32(1)

    f0 = @inbounds if0[pos]
    f1 = @inbounds if1234[pos0 * Int32(4) + Int32(1)]
    f2 = @inbounds if1234[pos0 * Int32(4) + Int32(2)]
    f3 = @inbounds if1234[pos0 * Int32(4) + Int32(3)]
    f4 = @inbounds if1234[pos0 * Int32(4) + Int32(4)]
    f5 = @inbounds if5678[pos0 * Int32(4) + Int32(1)]
    f6 = @inbounds if5678[pos0 * Int32(4) + Int32(2)]
    f7 = @inbounds if5678[pos0 * Int32(4) + Int32(3)]
    f8 = @inbounds if5678[pos0 * Int32(4) + Int32(4)]

    e0 = f0
    e1 = f1; e2 = f2; e3 = f3; e4 = f4
    e5 = f5; e6 = f6; e7 = f7; e8 = f8
    if @inbounds(types[pos]) == UInt8(1)
        e1 = f3; e2 = f4; e3 = f1; e4 = f2
        e5 = f7; e6 = f8; e7 = f5; e8 = f6
    else
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
        e0 = (1.0 - omega) * f0 + omega * ced(rho, weights[1], 0.0, 0.0, ux, uy)
        e1 = (1.0 - omega) * f1 + omega * ced(rho, weights[2], 1.0, 0.0, ux, uy)
        e2 = (1.0 - omega) * f2 + omega * ced(rho, weights[3], 0.0, 1.0, ux, uy)
        e3 = (1.0 - omega) * f3 + omega * ced(rho, weights[4], -1.0, 0.0, ux, uy)
        e4 = (1.0 - omega) * f4 + omega * ced(rho, weights[5], 0.0, -1.0, ux, uy)
        e5 = (1.0 - omega) * f5 + omega * ced(rho, weights[6], 1.0, 1.0, ux, uy)
        e6 = (1.0 - omega) * f6 + omega * ced(rho, weights[7], -1.0, 1.0, ux, uy)
        e7 = (1.0 - omega) * f7 + omega * ced(rho, weights[8], -1.0, -1.0, ux, uy)
        e8 = (1.0 - omega) * f8 + omega * ced(rho, weights[9], 1.0, -1.0, ux, uy)
    end

    if x > 0 && x < width - Int32(1) && y > 0 && y < height - Int32(1)
        @inbounds of0[pos] = e0
        @inbounds of1234[(pos0 + Int32(1)) * Int32(4) + Int32(1)] = e1
        @inbounds of1234[(pos0 + width) * Int32(4) + Int32(2)] = e2
        @inbounds of1234[(pos0 - Int32(1)) * Int32(4) + Int32(3)] = e3
        @inbounds of1234[(pos0 - width) * Int32(4) + Int32(4)] = e4
        @inbounds of5678[(pos0 + width + Int32(1)) * Int32(4) + Int32(1)] = e5
        @inbounds of5678[(pos0 + width - Int32(1)) * Int32(4) + Int32(2)] = e6
        @inbounds of5678[(pos0 - width - Int32(1)) * Int32(4) + Int32(3)] = e7
        @inbounds of5678[(pos0 - width + Int32(1)) * Int32(4) + Int32(4)] = e8
    end
    return
end


using CUDA
using Printf
using Random

const D = 64
const TOLERANCE = 1.0e-3

@inline idx3(i, j, k, n, m) =
    Int64(k) * Int64(m) * Int64(n) + Int64(j) * Int64(n) + Int64(i) + 1

function hypterm_1!(flux0, flux1, flux2, flux3, flux4, cons1, cons2, cons3, cons4,
                    q1, q2, q3, q4, dxinv0::Float64, dxinv1::Float64, dxinv2::Float64,
                    ldim::Int32, mdim::Int32, ndim::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if i >= 4 && j >= 4 && k >= 4 && i <= ndim - 5 && j <= ndim - 5 && k <= ndim - 5
        p = idx3(i, j, k, ndim, mdim)
        @inbounds begin
            flux0[p] = -((0.8 * (cons1[idx3(i+1,j,k,ndim,mdim)] - cons1[idx3(i-1,j,k,ndim,mdim)]) -
                          0.2 * (cons1[idx3(i+2,j,k,ndim,mdim)] - cons1[idx3(i-2,j,k,ndim,mdim)]) +
                          0.038 * (cons1[idx3(i+3,j,k,ndim,mdim)] - cons1[idx3(i-3,j,k,ndim,mdim)]) -
                          0.0035 * (cons1[idx3(i+4,j,k,ndim,mdim)] - cons1[idx3(i-4,j,k,ndim,mdim)])) * dxinv0)
            flux1[p] = -((0.8 * (cons1[idx3(i+1,j,k,ndim,mdim)] * q1[idx3(i+1,j,k,ndim,mdim)] - cons1[idx3(i-1,j,k,ndim,mdim)] * q1[idx3(i-1,j,k,ndim,mdim)] + q4[idx3(i+1,j,k,ndim,mdim)] - q4[idx3(i-1,j,k,ndim,mdim)]) -
                          0.2 * (cons1[idx3(i+2,j,k,ndim,mdim)] * q1[idx3(i+2,j,k,ndim,mdim)] - cons1[idx3(i-2,j,k,ndim,mdim)] * q1[idx3(i-2,j,k,ndim,mdim)] + q4[idx3(i+2,j,k,ndim,mdim)] - q4[idx3(i-2,j,k,ndim,mdim)]) +
                          0.038 * (cons1[idx3(i+3,j,k,ndim,mdim)] * q1[idx3(i+3,j,k,ndim,mdim)] - cons1[idx3(i-3,j,k,ndim,mdim)] * q1[idx3(i-3,j,k,ndim,mdim)] + q4[idx3(i+3,j,k,ndim,mdim)] - q4[idx3(i-3,j,k,ndim,mdim)]) -
                          0.0035 * (cons1[idx3(i+4,j,k,ndim,mdim)] * q1[idx3(i+4,j,k,ndim,mdim)] - cons1[idx3(i-4,j,k,ndim,mdim)] * q1[idx3(i-4,j,k,ndim,mdim)] + q4[idx3(i+4,j,k,ndim,mdim)] - q4[idx3(i-4,j,k,ndim,mdim)])) * dxinv0)
            flux2[p] = -((0.8 * (cons2[idx3(i+1,j,k,ndim,mdim)] * q1[idx3(i+1,j,k,ndim,mdim)] - cons2[idx3(i-1,j,k,ndim,mdim)] * q1[idx3(i-1,j,k,ndim,mdim)]) -
                          0.2 * (cons2[idx3(i+2,j,k,ndim,mdim)] * q1[idx3(i+2,j,k,ndim,mdim)] - cons2[idx3(i-2,j,k,ndim,mdim)] * q1[idx3(i-2,j,k,ndim,mdim)]) +
                          0.038 * (cons2[idx3(i+3,j,k,ndim,mdim)] * q1[idx3(i+3,j,k,ndim,mdim)] - cons2[idx3(i-3,j,k,ndim,mdim)] * q1[idx3(i-3,j,k,ndim,mdim)]) -
                          0.0035 * (cons2[idx3(i+4,j,k,ndim,mdim)] * q1[idx3(i+4,j,k,ndim,mdim)] - cons2[idx3(i-4,j,k,ndim,mdim)] * q1[idx3(i-4,j,k,ndim,mdim)])) * dxinv0)
            flux3[p] = -((0.8 * (cons3[idx3(i+1,j,k,ndim,mdim)] * q1[idx3(i+1,j,k,ndim,mdim)] - cons3[idx3(i-1,j,k,ndim,mdim)] * q1[idx3(i-1,j,k,ndim,mdim)]) -
                          0.2 * (cons3[idx3(i+2,j,k,ndim,mdim)] * q1[idx3(i+2,j,k,ndim,mdim)] - cons3[idx3(i-2,j,k,ndim,mdim)] * q1[idx3(i-2,j,k,ndim,mdim)]) +
                          0.038 * (cons3[idx3(i+3,j,k,ndim,mdim)] * q1[idx3(i+3,j,k,ndim,mdim)] - cons3[idx3(i-3,j,k,ndim,mdim)] * q1[idx3(i-3,j,k,ndim,mdim)]) -
                          0.0035 * (cons3[idx3(i+4,j,k,ndim,mdim)] * q1[idx3(i+4,j,k,ndim,mdim)] - cons3[idx3(i-4,j,k,ndim,mdim)] * q1[idx3(i-4,j,k,ndim,mdim)])) * dxinv0)
            flux4[p] = -((0.8 * (cons4[idx3(i+1,j,k,ndim,mdim)] * q1[idx3(i+1,j,k,ndim,mdim)] - cons4[idx3(i-1,j,k,ndim,mdim)] * q1[idx3(i-1,j,k,ndim,mdim)] + q4[idx3(i+1,j,k,ndim,mdim)] * q1[idx3(i+1,j,k,ndim,mdim)] - q4[idx3(i-1,j,k,ndim,mdim)] * q1[idx3(i-1,j,k,ndim,mdim)]) -
                          0.2 * (cons4[idx3(i+2,j,k,ndim,mdim)] * q1[idx3(i+2,j,k,ndim,mdim)] - cons4[idx3(i-2,j,k,ndim,mdim)] * q1[idx3(i-2,j,k,ndim,mdim)] + q4[idx3(i+2,j,k,ndim,mdim)] * q1[idx3(i+2,j,k,ndim,mdim)] - q4[idx3(i-2,j,k,ndim,mdim)] * q1[idx3(i-2,j,k,ndim,mdim)]) +
                          0.038 * (cons4[idx3(i+3,j,k,ndim,mdim)] * q1[idx3(i+3,j,k,ndim,mdim)] - cons4[idx3(i-3,j,k,ndim,mdim)] * q1[idx3(i-3,j,k,ndim,mdim)] + q4[idx3(i+3,j,k,ndim,mdim)] * q1[idx3(i+3,j,k,ndim,mdim)] - q4[idx3(i-3,j,k,ndim,mdim)] * q1[idx3(i-3,j,k,ndim,mdim)]) -
                          0.0035 * (cons4[idx3(i+4,j,k,ndim,mdim)] * q1[idx3(i+4,j,k,ndim,mdim)] - cons4[idx3(i-4,j,k,ndim,mdim)] * q1[idx3(i-4,j,k,ndim,mdim)] + q4[idx3(i+4,j,k,ndim,mdim)] * q1[idx3(i+4,j,k,ndim,mdim)] - q4[idx3(i-4,j,k,ndim,mdim)] * q1[idx3(i-4,j,k,ndim,mdim)])) * dxinv0)
        end
    end
    return
end

function hypterm_2!(flux0, flux1, flux2, flux3, flux4, cons1, cons2, cons3, cons4,
                    q1, q2, q3, q4, dxinv0::Float64, dxinv1::Float64, dxinv2::Float64,
                    ldim::Int32, mdim::Int32, ndim::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if i >= 4 && j >= 4 && k >= 4 && i <= ndim - 5 && j <= ndim - 5 && k <= ndim - 5
        p = idx3(i, j, k, ndim, mdim)
        @inbounds begin
            flux0[p] -= (0.8 * (cons2[idx3(i,j+1,k,ndim,mdim)] - cons2[idx3(i,j-1,k,ndim,mdim)]) - 0.2 * (cons2[idx3(i,j+2,k,ndim,mdim)] - cons2[idx3(i,j-2,k,ndim,mdim)]) + 0.038 * (cons2[idx3(i,j+3,k,ndim,mdim)] - cons2[idx3(i,j-3,k,ndim,mdim)]) - 0.0035 * (cons2[idx3(i,j+4,k,ndim,mdim)] - cons2[idx3(i,j-4,k,ndim,mdim)])) * dxinv1
            flux1[p] -= (0.8 * (cons1[idx3(i,j+1,k,ndim,mdim)] * q2[idx3(i,j+1,k,ndim,mdim)] - cons1[idx3(i,j-1,k,ndim,mdim)] * q2[idx3(i,j-1,k,ndim,mdim)]) - 0.2 * (cons1[idx3(i,j+2,k,ndim,mdim)] * q2[idx3(i,j+2,k,ndim,mdim)] - cons1[idx3(i,j-2,k,ndim,mdim)] * q2[idx3(i,j-2,k,ndim,mdim)]) + 0.038 * (cons1[idx3(i,j+3,k,ndim,mdim)] * q2[idx3(i,j+3,k,ndim,mdim)] - cons1[idx3(i,j-3,k,ndim,mdim)] * q2[idx3(i,j-3,k,ndim,mdim)]) - 0.0035 * (cons1[idx3(i,j+4,k,ndim,mdim)] * q2[idx3(i,j+4,k,ndim,mdim)] - cons1[idx3(i,j-4,k,ndim,mdim)] * q2[idx3(i,j-4,k,ndim,mdim)])) * dxinv1
            flux2[p] -= (0.8 * (cons2[idx3(i,j+1,k,ndim,mdim)] * q2[idx3(i,j+1,k,ndim,mdim)] - cons2[idx3(i,j-1,k,ndim,mdim)] * q2[idx3(i,j-1,k,ndim,mdim)] + q4[idx3(i,j+1,k,ndim,mdim)] - q4[idx3(i,j-1,k,ndim,mdim)]) - 0.2 * (cons2[idx3(i,j+2,k,ndim,mdim)] * q2[idx3(i,j+2,k,ndim,mdim)] - cons2[idx3(i,j-2,k,ndim,mdim)] * q2[idx3(i,j-2,k,ndim,mdim)] + q4[idx3(i,j+2,k,ndim,mdim)] - q4[idx3(i,j-2,k,ndim,mdim)]) + 0.038 * (cons2[idx3(i,j+3,k,ndim,mdim)] * q2[idx3(i,j+3,k,ndim,mdim)] - cons2[idx3(i,j-3,k,ndim,mdim)] * q2[idx3(i,j-3,k,ndim,mdim)] + q4[idx3(i,j+3,k,ndim,mdim)] - q4[idx3(i,j-3,k,ndim,mdim)]) - 0.0035 * (cons2[idx3(i,j+4,k,ndim,mdim)] * q2[idx3(i,j+4,k,ndim,mdim)] - cons2[idx3(i,j-4,k,ndim,mdim)] * q2[idx3(i,j-4,k,ndim,mdim)] + q4[idx3(i,j+4,k,ndim,mdim)] - q4[idx3(i,j-4,k,ndim,mdim)])) * dxinv1
            flux3[p] -= (0.8 * (cons3[idx3(i,j+1,k,ndim,mdim)] * q2[idx3(i,j+1,k,ndim,mdim)] - cons3[idx3(i,j-1,k,ndim,mdim)] * q2[idx3(i,j-1,k,ndim,mdim)]) - 0.2 * (cons3[idx3(i,j+2,k,ndim,mdim)] * q2[idx3(i,j+2,k,ndim,mdim)] - cons3[idx3(i,j-2,k,ndim,mdim)] * q2[idx3(i,j-2,k,ndim,mdim)]) + 0.038 * (cons3[idx3(i,j+3,k,ndim,mdim)] * q2[idx3(i,j+3,k,ndim,mdim)] - cons3[idx3(i,j-3,k,ndim,mdim)] * q2[idx3(i,j-3,k,ndim,mdim)]) - 0.0035 * (cons3[idx3(i,j+4,k,ndim,mdim)] * q2[idx3(i,j+4,k,ndim,mdim)] - cons3[idx3(i,j-4,k,ndim,mdim)] * q2[idx3(i,j-4,k,ndim,mdim)])) * dxinv1
            flux4[p] -= (0.8 * (cons4[idx3(i,j,k+1,ndim,mdim)] * q3[idx3(i,j,k+1,ndim,mdim)] - cons4[idx3(i,j,k-1,ndim,mdim)] * q3[idx3(i,j,k-1,ndim,mdim)] + q4[idx3(i,j,k+1,ndim,mdim)] * q3[idx3(i,j,k+1,ndim,mdim)] - q4[idx3(i,j,k-1,ndim,mdim)] * q3[idx3(i,j,k-1,ndim,mdim)]) - 0.2 * (cons4[idx3(i,j,k+2,ndim,mdim)] * q3[idx3(i,j,k+2,ndim,mdim)] - cons4[idx3(i,j,k-2,ndim,mdim)] * q3[idx3(i,j,k-2,ndim,mdim)] + q4[idx3(i,j,k+2,ndim,mdim)] * q3[idx3(i,j,k+2,ndim,mdim)] - q4[idx3(i,j,k-2,ndim,mdim)] * q3[idx3(i,j,k-2,ndim,mdim)]) + 0.038 * (cons4[idx3(i,j,k+3,ndim,mdim)] * q3[idx3(i,j,k+3,ndim,mdim)] - cons4[idx3(i,j,k-3,ndim,mdim)] * q3[idx3(i,j,k-3,ndim,mdim)] + q4[idx3(i,j,k+3,ndim,mdim)] * q3[idx3(i,j,k+3,ndim,mdim)] - q4[idx3(i,j,k-3,ndim,mdim)]) - 0.0035 * (cons4[idx3(i,j,k+4,ndim,mdim)] * q3[idx3(i,j,k+4,ndim,mdim)] - cons4[idx3(i,j,k-4,ndim,mdim)] * q3[idx3(i,j,k-4,ndim,mdim)] + q4[idx3(i,j,k+4,ndim,mdim)] * q3[idx3(i,j,k+4,ndim,mdim)] - q4[idx3(i,j,k-4,ndim,mdim)] * q3[idx3(i,j,k-4,ndim,mdim)])) * dxinv2
        end
    end
    return
end

function hypterm_3!(flux0, flux1, flux2, flux3, flux4, cons1, cons2, cons3, cons4,
                    q1, q2, q3, q4, dxinv0::Float64, dxinv1::Float64, dxinv2::Float64,
                    ldim::Int32, mdim::Int32, ndim::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if i >= 4 && j >= 4 && k >= 4 && i <= ndim - 5 && j <= ndim - 5 && k <= ndim - 5
        p = idx3(i, j, k, ndim, mdim)
        @inbounds begin
            flux0[p] -= (0.8 * (cons3[idx3(i,j,k+1,ndim,mdim)] - cons3[idx3(i,j,k-1,ndim,mdim)]) - 0.2 * (cons3[idx3(i,j,k+2,ndim,mdim)] - cons3[idx3(i,j,k-2,ndim,mdim)]) + 0.038 * (cons3[idx3(i,j,k+3,ndim,mdim)] - cons3[idx3(i,j,k-3,ndim,mdim)]) - 0.0035 * (cons3[idx3(i,j,k+4,ndim,mdim)] - cons3[idx3(i,j,k-4,ndim,mdim)])) * dxinv2
            flux1[p] -= (0.8 * (cons1[idx3(i,j,k+1,ndim,mdim)] * q3[idx3(i,j,k+1,ndim,mdim)] - cons1[idx3(i,j,k-1,ndim,mdim)] * q3[idx3(i,j,k-1,ndim,mdim)]) - 0.2 * (cons1[idx3(i,j,k+2,ndim,mdim)] * q3[idx3(i,j,k+2,ndim,mdim)] - cons1[idx3(i,j,k-2,ndim,mdim)] * q3[idx3(i,j,k-2,ndim,mdim)]) + 0.038 * (cons1[idx3(i,j,k+3,ndim,mdim)] * q3[idx3(i,j,k+3,ndim,mdim)] - cons1[idx3(i,j,k-3,ndim,mdim)] * q3[idx3(i,j,k-3,ndim,mdim)]) - 0.0035 * (cons1[idx3(i,j,k+4,ndim,mdim)] * q3[idx3(i,j,k+4,ndim,mdim)] - cons1[idx3(i,j,k-4,ndim,mdim)] * q3[idx3(i,j,k-4,ndim,mdim)])) * dxinv2
            flux2[p] -= (0.8 * (cons2[idx3(i,j,k+1,ndim,mdim)] * q3[idx3(i,j,k+1,ndim,mdim)] - cons2[idx3(i,j,k-1,ndim,mdim)] * q3[idx3(i,j,k-1,ndim,mdim)]) - 0.2 * (cons2[idx3(i,j,k+2,ndim,mdim)] * q3[idx3(i,j,k+2,ndim,mdim)] - cons2[idx3(i,j,k-2,ndim,mdim)] * q3[idx3(i,j,k-2,ndim,mdim)]) + 0.038 * (cons2[idx3(i,j,k+3,ndim,mdim)] * q3[idx3(i,j,k+3,ndim,mdim)] - cons2[idx3(i,j,k-3,ndim,mdim)] * q3[idx3(i,j,k-3,ndim,mdim)]) - 0.0035 * (cons2[idx3(i,j,k+4,ndim,mdim)] * q3[idx3(i,j,k+4,ndim,mdim)] - cons2[idx3(i,j,k-4,ndim,mdim)] * q3[idx3(i,j,k-4,ndim,mdim)])) * dxinv2
            flux3[p] -= (0.8 * (cons3[idx3(i,j,k+1,ndim,mdim)] * q3[idx3(i,j,k+1,ndim,mdim)] - cons3[idx3(i,j,k-1,ndim,mdim)] * q3[idx3(i,j,k-1,ndim,mdim)] + q4[idx3(i,j,k+1,ndim,mdim)] - q4[idx3(i,j,k-1,ndim,mdim)]) - 0.2 * (cons3[idx3(i,j,k+2,ndim,mdim)] * q3[idx3(i,j,k+2,ndim,mdim)] - cons3[idx3(i,j,k-2,ndim,mdim)] * q3[idx3(i,j,k-2,ndim,mdim)] + q4[idx3(i,j,k+2,ndim,mdim)] - q4[idx3(i,j,k-2,ndim,mdim)]) + 0.038 * (cons3[idx3(i,j,k+3,ndim,mdim)] * q3[idx3(i,j,k+3,ndim,mdim)] - cons3[idx3(i,j,k-3,ndim,mdim)] * q3[idx3(i,j,k-3,ndim,mdim)] + q4[idx3(i,j,k+3,ndim,mdim)] - q4[idx3(i,j,k-3,ndim,mdim)]) - 0.0035 * (cons3[idx3(i,j,k+4,ndim,mdim)] * q3[idx3(i,j,k+4,ndim,mdim)] - cons3[idx3(i,j,k-4,ndim,mdim)] * q3[idx3(i,j,k-4,ndim,mdim)] + q4[idx3(i,j,k+4,ndim,mdim)] - q4[idx3(i,j,k-4,ndim,mdim)])) * dxinv2
            flux4[p] -= (0.8 * (cons4[idx3(i,j+1,k,ndim,mdim)] * q2[idx3(i,j+1,k,ndim,mdim)] - cons4[idx3(i,j-1,k,ndim,mdim)] * q2[idx3(i,j-1,k,ndim,mdim)] + q4[idx3(i,j+1,k,ndim,mdim)] * q2[idx3(i,j+1,k,ndim,mdim)] - q4[idx3(i,j-1,k,ndim,mdim)] * q2[idx3(i,j-1,k,ndim,mdim)]) - 0.2 * (cons4[idx3(i,j+2,k,ndim,mdim)] * q2[idx3(i,j+2,k,ndim,mdim)] - cons4[idx3(i,j-2,k,ndim,mdim)] * q2[idx3(i,j-2,k,ndim,mdim)] + q4[idx3(i,j+2,k,ndim,mdim)] * q2[idx3(i,j+2,k,ndim,mdim)] - q4[idx3(i,j-2,k,ndim,mdim)] * q2[idx3(i,j-2,k,ndim,mdim)]) + 0.038 * (cons4[idx3(i,j+3,k,ndim,mdim)] * q2[idx3(i,j+3,k,ndim,mdim)] - cons4[idx3(i,j-3,k,ndim,mdim)] * q2[idx3(i,j-3,k,ndim,mdim)] + q4[idx3(i,j+3,k,ndim,mdim)] * q2[idx3(i,j+3,k,ndim,mdim)] - q4[idx3(i,j-3,k,ndim,mdim)] * q2[idx3(i,j-3,k,ndim,mdim)]) - 0.0035 * (cons4[idx3(i,j+4,k,ndim,mdim)] * q2[idx3(i,j+4,k,ndim,mdim)] - cons4[idx3(i,j-4,k,ndim,mdim)] * q2[idx3(i,j-4,k,ndim,mdim)] + q4[idx3(i,j+4,k,ndim,mdim)] * q2[idx3(i,j+4,k,ndim,mdim)] - q4[idx3(i,j-4,k,ndim,mdim)] * q2[idx3(i,j-4,k,ndim,mdim)])) * dxinv1
        end
    end
    return
end

function reference!(flux, cons, q, dxinv)
    flux0, flux1, flux2, flux3, flux4 = flux
    cons1, cons2, cons3, cons4 = cons
    q1, q2, q3, q4 = q
    n = D
    @inbounds for k in 5:(n-4), j in 5:(n-4), i in 5:(n-4)
        flux0[i,j,k] = -((0.8 * (cons1[i+1,j,k] - cons1[i-1,j,k]) - 0.2 * (cons1[i+2,j,k] - cons1[i-2,j,k]) + 0.038 * (cons1[i+3,j,k] - cons1[i-3,j,k]) - 0.0035 * (cons1[i+4,j,k] - cons1[i-4,j,k])) * dxinv[1])
        flux0[i,j,k] -= (0.8 * (cons2[i,j+1,k] - cons2[i,j-1,k]) - 0.2 * (cons2[i,j+2,k] - cons2[i,j-2,k]) + 0.038 * (cons2[i,j+3,k] - cons2[i,j-3,k]) - 0.0035 * (cons2[i,j+4,k] - cons2[i,j-4,k])) * dxinv[2]
        flux0[i,j,k] -= (0.8 * (cons3[i,j,k+1] - cons3[i,j,k-1]) - 0.2 * (cons3[i,j,k+2] - cons3[i,j,k-2]) + 0.038 * (cons3[i,j,k+3] - cons3[i,j,k-3]) - 0.0035 * (cons3[i,j,k+4] - cons3[i,j,k-4])) * dxinv[3]
        flux1[i,j,k] = -((0.8 * (cons1[i+1,j,k] * q1[i+1,j,k] - cons1[i-1,j,k] * q1[i-1,j,k] + q4[i+1,j,k] - q4[i-1,j,k]) - 0.2 * (cons1[i+2,j,k] * q1[i+2,j,k] - cons1[i-2,j,k] * q1[i-2,j,k] + q4[i+2,j,k] - q4[i-2,j,k]) + 0.038 * (cons1[i+3,j,k] * q1[i+3,j,k] - cons1[i-3,j,k] * q1[i-3,j,k] + q4[i+3,j,k] - q4[i-3,j,k]) - 0.0035 * (cons1[i+4,j,k] * q1[i+4,j,k] - cons1[i-4,j,k] * q1[i-4,j,k] + q4[i+4,j,k] - q4[i-4,j,k])) * dxinv[1])
        flux1[i,j,k] -= (0.8 * (cons1[i,j+1,k] * q2[i,j+1,k] - cons1[i,j-1,k] * q2[i,j-1,k]) - 0.2 * (cons1[i,j+2,k] * q2[i,j+2,k] - cons1[i,j-2,k] * q2[i,j-2,k]) + 0.038 * (cons1[i,j+3,k] * q2[i,j+3,k] - cons1[i,j-3,k] * q2[i,j-3,k]) - 0.0035 * (cons1[i,j+4,k] * q2[i,j+4,k] - cons1[i,j-4,k] * q2[i,j-4,k])) * dxinv[2]
        flux1[i,j,k] -= (0.8 * (cons1[i,j,k+1] * q3[i,j,k+1] - cons1[i,j,k-1] * q3[i,j,k-1]) - 0.2 * (cons1[i,j,k+2] * q3[i,j,k+2] - cons1[i,j,k-2] * q3[i,j,k-2]) + 0.038 * (cons1[i,j,k+3] * q3[i,j,k+3] - cons1[i,j,k-3] * q3[i,j,k-3]) - 0.0035 * (cons1[i,j,k+4] * q3[i,j,k+4] - cons1[i,j,k-4] * q3[i,j,k-4])) * dxinv[3]
        flux2[i,j,k] = -((0.8 * (cons2[i+1,j,k] * q1[i+1,j,k] - cons2[i-1,j,k] * q1[i-1,j,k]) - 0.2 * (cons2[i+2,j,k] * q1[i+2,j,k] - cons2[i-2,j,k] * q1[i-2,j,k]) + 0.038 * (cons2[i+3,j,k] * q1[i+3,j,k] - cons2[i-3,j,k] * q1[i-3,j,k]) - 0.0035 * (cons2[i+4,j,k] * q1[i+4,j,k] - cons2[i-4,j,k] * q1[i-4,j,k])) * dxinv[1])
        flux2[i,j,k] -= (0.8 * (cons2[i,j+1,k] * q2[i,j+1,k] - cons2[i,j-1,k] * q2[i,j-1,k] + q4[i,j+1,k] - q4[i,j-1,k]) - 0.2 * (cons2[i,j+2,k] * q2[i,j+2,k] - cons2[i,j-2,k] * q2[i,j-2,k] + q4[i,j+2,k] - q4[i,j-2,k]) + 0.038 * (cons2[i,j+3,k] * q2[i,j+3,k] - cons2[i,j-3,k] * q2[i,j-3,k] + q4[i,j+3,k] - q4[i,j-3,k]) - 0.0035 * (cons2[i,j+4,k] * q2[i,j+4,k] - cons2[i,j-4,k] * q2[i,j-4,k] + q4[i,j+4,k] - q4[i,j-4,k])) * dxinv[2]
        flux2[i,j,k] -= (0.8 * (cons2[i,j,k+1] * q3[i,j,k+1] - cons2[i,j,k-1] * q3[i,j,k-1]) - 0.2 * (cons2[i,j,k+2] * q3[i,j,k+2] - cons2[i,j,k-2] * q3[i,j,k-2]) + 0.038 * (cons2[i,j,k+3] * q3[i,j,k+3] - cons2[i,j,k-3] * q3[i,j,k-3]) - 0.0035 * (cons2[i,j,k+4] * q3[i,j,k+4] - cons2[i,j,k-4] * q3[i,j,k-4])) * dxinv[3]
        flux3[i,j,k] = -((0.8 * (cons3[i+1,j,k] * q1[i+1,j,k] - cons3[i-1,j,k] * q1[i-1,j,k]) - 0.2 * (cons3[i+2,j,k] * q1[i+2,j,k] - cons3[i-2,j,k] * q1[i-2,j,k]) + 0.038 * (cons3[i+3,j,k] * q1[i+3,j,k] - cons3[i-3,j,k] * q1[i-3,j,k]) - 0.0035 * (cons3[i+4,j,k] * q1[i+4,j,k] - cons3[i-4,j,k] * q1[i-4,j,k])) * dxinv[1])
        flux3[i,j,k] -= (0.8 * (cons3[i,j+1,k] * q2[i,j+1,k] - cons3[i,j-1,k] * q2[i,j-1,k]) - 0.2 * (cons3[i,j+2,k] * q2[i,j+2,k] - cons3[i,j-2,k] * q2[i,j-2,k]) + 0.038 * (cons3[i,j+3,k] * q2[i,j+3,k] - cons3[i,j-3,k] * q2[i,j-3,k]) - 0.0035 * (cons3[i,j+4,k] * q2[i,j+4,k] - cons3[i,j-4,k] * q2[i,j-4,k])) * dxinv[2]
        flux3[i,j,k] -= (0.8 * (cons3[i,j,k+1] * q3[i,j,k+1] - cons3[i,j,k-1] * q3[i,j,k-1] + q4[i,j,k+1] - q4[i,j,k-1]) - 0.2 * (cons3[i,j,k+2] * q3[i,j,k+2] - cons3[i,j,k-2] * q3[i,j,k-2] + q4[i,j,k+2] - q4[i,j,k-2]) + 0.038 * (cons3[i,j,k+3] * q3[i,j,k+3] - cons3[i,j,k-3] * q3[i,j,k-3] + q4[i,j,k+3] - q4[i,j,k-3]) - 0.0035 * (cons3[i,j,k+4] * q3[i,j,k+4] - cons3[i,j,k-4] * q3[i,j,k-4] + q4[i,j,k+4] - q4[i,j,k-4])) * dxinv[3]
        flux4[i,j,k] = -((0.8 * (cons4[i+1,j,k] * q1[i+1,j,k] - cons4[i-1,j,k] * q1[i-1,j,k] + q4[i+1,j,k] * q1[i+1,j,k] - q4[i-1,j,k] * q1[i-1,j,k]) - 0.2 * (cons4[i+2,j,k] * q1[i+2,j,k] - cons4[i-2,j,k] * q1[i-2,j,k] + q4[i+2,j,k] * q1[i+2,j,k] - q4[i-2,j,k] * q1[i-2,j,k]) + 0.038 * (cons4[i+3,j,k] * q1[i+3,j,k] - cons4[i-3,j,k] * q1[i-3,j,k] + q4[i+3,j,k] * q1[i+3,j,k] - q4[i-3,j,k] * q1[i-3,j,k]) - 0.0035 * (cons4[i+4,j,k] * q1[i+4,j,k] - cons4[i-4,j,k] * q1[i-4,j,k] + q4[i+4,j,k] * q1[i+4,j,k] - q4[i-4,j,k] * q1[i-4,j,k])) * dxinv[1])
        flux4[i,j,k] -= (0.8 * (cons4[i,j,k+1] * q3[i,j,k+1] - cons4[i,j,k-1] * q3[i,j,k-1] + q4[i,j,k+1] * q3[i,j,k+1] - q4[i,j,k-1] * q3[i,j,k-1]) - 0.2 * (cons4[i,j,k+2] * q3[i,j,k+2] - cons4[i,j,k-2] * q3[i,j,k-2] + q4[i,j,k+2] * q3[i,j,k+2] - q4[i,j,k-2] * q3[i,j,k-2]) + 0.038 * (cons4[i,j,k+3] * q3[i,j,k+3] - cons4[i,j,k-3] * q3[i,j,k-3] + q4[i,j,k+3] * q3[i,j,k+3] - q4[i,j,k-3] * q3[i,j,k-3]) - 0.0035 * (cons4[i,j,k+4] * q3[i,j,k+4] - cons4[i,j,k-4] * q3[i,j,k-4] + q4[i,j,k+4] * q3[i,j,k+4] - q4[i,j,k-4] * q3[i,j,k-4])) * dxinv[3]
        flux4[i,j,k] -= (0.8 * (cons4[i,j+1,k] * q2[i,j+1,k] - cons4[i,j-1,k] * q2[i,j-1,k] + q4[i,j+1,k] * q2[i,j+1,k] - q4[i,j-1,k] * q2[i,j-1,k]) - 0.2 * (cons4[i,j+2,k] * q2[i,j+2,k] - cons4[i,j-2,k] * q2[i,j-2,k] + q4[i,j+2,k] * q2[i,j+2,k] - q4[i,j-2,k] * q2[i,j-2,k]) + 0.038 * (cons4[i,j+3,k] * q2[i,j+3,k] - cons4[i,j-3,k] * q2[i,j-3,k] + q4[i,j+3,k] * q2[i,j+3,k] - q4[i,j-3,k] * q2[i,j-3,k]) - 0.0035 * (cons4[i,j+4,k] * q2[i,j+4,k] - cons4[i,j-4,k] * q2[i,j-4,k] + q4[i,j+4,k] * q2[i,j+4,k] - q4[i,j-4,k] * q2[i,j-4,k])) * dxinv[2]
    end
end

function run_offload!(flux, cons, q, dxinv, repeat)
    dflux = map(CuArray, flux)
    dcons = map(CuArray, cons)
    dq = map(CuArray, q)
    threads = (16, 4, 4)
    blocks = (cld(D, 16), cld(D, 4), cld(D, 4))
    t = zeros(Int, 3)
    for _ in 1:repeat
        for a in dflux
            CUDA.fill!(a, 0.0)
        end
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks hypterm_1!(dflux..., dcons..., dq..., dxinv..., Int32(D), Int32(D), Int32(D))
        CUDA.synchronize()
        t[1] += time_ns() - start
        start = time_ns()
        @cuda threads=threads blocks=blocks hypterm_2!(dflux..., dcons..., dq..., dxinv..., Int32(D), Int32(D), Int32(D))
        CUDA.synchronize()
        t[2] += time_ns() - start
        start = time_ns()
        @cuda threads=threads blocks=blocks hypterm_3!(dflux..., dcons..., dq..., dxinv..., Int32(D), Int32(D), Int32(D))
        CUDA.synchronize()
        t[3] += time_ns() - start
    end
    @printf("Average kernel execution time (k1): %f (ms)\n", t[1] * 1.0e-6 / repeat)
    @printf("Average kernel execution time (k2): %f (ms)\n", t[2] * 1.0e-6 / repeat)
    @printf("Average kernel execution time (k3): %f (ms)\n", t[3] * 1.0e-6 / repeat)
    return map(Array, dflux)
end

function check_error_3d(output, reference)
    err = 0.0
    sum_output = 0.0
    count = 0
    for k in 5:(D-4), j in 5:(D-4), i in 5:(D-4)
        curr = abs(output[i,j,k] - reference[i,j,k])
        err += curr * curr
        sum_output += output[i,j,k]
        count += 1
    end
    @printf("checksum = %e\n", sum_output)
    return sqrt(err / count)
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    repeat > 0 || return 1
    rng = MersenneTwister(123)
    cons = [rand(rng, Float64, D, D, D) .+ 0.02121 for _ in 1:4]
    q = [rand(rng, Float64, D, D, D) .+ 0.02121 for _ in 1:4]
    flux = [zeros(Float64, D, D, D) for _ in 1:5]
    flux_gold = [zeros(Float64, D, D, D) for _ in 1:5]
    dxinv = (0.01, 0.02, 0.03)

    reference!(flux_gold, cons, q, dxinv)
    flux_out = run_offload!(flux, cons, q, dxinv, repeat)

    ok = true
    for n in 1:5
        @printf("Check flux_%d\n", n - 1)
        error = check_error_3d(flux_out[n], flux_gold[n])
        @printf("RMS Error : %e\n", error)
        ok &= error <= TOLERANCE
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

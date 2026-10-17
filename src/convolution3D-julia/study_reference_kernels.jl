using CUDA
using Printf
using Random

const TILE_WIDTH = Int32(16)

@inline function ii(n::Int32, c::Int32, h::Int32, w::Int32, C::Int32, Hin::Int32, Win::Int32)
    return ((n * C * Hin * Win) + (c * Hin * Win) + (h * Win) + w) + Int32(1)
end

@inline function wi(m::Int32, c::Int32, p::Int32, q::Int32, C::Int32, K::Int32)
    return ((m * C * K * K) + (c * K * K) + (p * K) + q) + Int32(1)
end

@inline function oi(n::Int32, m::Int32, h::Int32, w::Int32, M::Int32, Hout::Int32, Wout::Int32)
    return ((n * M * Hout * Wout) + (m * Hout * Wout) + (h * Wout) + w) + Int32(1)
end

function conv3d_s1_kernel!(x, w, y, C::Int32, M::Int32, K::Int32,
                           Hin::Int32, Win::Int32, Hout::Int32, Wout::Int32,
                           W_grid::Int32)
    n = blockIdx().x - Int32(1)
    m = blockIdx().y - Int32(1)
    z = blockIdx().z - Int32(1)
    h = (z ÷ W_grid) * TILE_WIDTH + threadIdx().y - Int32(1)
    col = (z % W_grid) * TILE_WIDTH + threadIdx().x - Int32(1)
    if h < Hout && col < Wout
        s = Float32(0)
        @inbounds for c in Int32(0):(C - Int32(1))
            for p in Int32(0):(K - Int32(1))
                for q in Int32(0):(K - Int32(1))
                    s += x[ii(n, c, h + p, col + q, C, Hin, Win)] *
                         w[wi(m, c, p, q, C, K)]
                end
            end
        end
        @inbounds y[oi(n, m, h, col, M, Hout, Wout)] = s
    end
    return
end

function conv3d_s2_kernel!(x, w, y, C::Int32, M::Int32, K::Int32,
                           Hin::Int32, Win::Int32, Hout::Int32, Wout::Int32,
                           W_grid::Int32)
    m = blockIdx().x - Int32(1)
    z = blockIdx().y - Int32(1)
    n = blockIdx().z - Int32(1)
    h = (z ÷ W_grid) * TILE_WIDTH + threadIdx().y - Int32(1)
    col = (z % W_grid) * TILE_WIDTH + threadIdx().x - Int32(1)
    if h < Hout && col < Wout
        s = Float32(0)
        @inbounds for c in Int32(0):(C - Int32(1))
            for p in Int32(0):(K - Int32(1))
                for q in Int32(0):(K - Int32(1))
                    s += x[ii(n, c, h + p, col + q, C, Hin, Win)] *
                         w[wi(m, c, p, q, C, K)]
                end
            end
        end
        @inbounds y[oi(n, m, h, col, M, Hout, Wout)] = s
    end
    return
end

function conv3d_s3_kernel!(x, w, y, C::Int32, M::Int32, K::Int32,
                           Hin::Int32, Win::Int32, Hout::Int32, Wout::Int32,
                           W_grid::Int32)
    z = blockIdx().x - Int32(1)
    n = blockIdx().y - Int32(1)
    m = blockIdx().z - Int32(1)
    h = (z ÷ W_grid) * TILE_WIDTH + threadIdx().y - Int32(1)
    col = (z % W_grid) * TILE_WIDTH + threadIdx().x - Int32(1)
    if h < Hout && col < Wout
        s = Float32(0)
        @inbounds for c in Int32(0):(C - Int32(1))
            for p in Int32(0):(K - Int32(1))
                for q in Int32(0):(K - Int32(1))
                    s += x[ii(n, c, h + p, col + q, C, Hin, Win)] *
                         w[wi(m, c, p, q, C, K)]
                end
            end
        end
        @inbounds y[oi(n, m, h, col, M, Hout, Wout)] = s
    end
    return
end


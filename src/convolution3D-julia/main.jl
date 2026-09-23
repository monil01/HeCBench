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

function reference!(x, w, y, N::Int32, M::Int32, C::Int32, K::Int32,
                    Hin::Int32, Win::Int32, Hout::Int32, Wout::Int32)
    @inbounds for n in Int32(0):(N - Int32(1))
        for m in Int32(0):(M - Int32(1))
            for h in Int32(0):(Hout - Int32(1))
                for col in Int32(0):(Wout - Int32(1))
                    s = Float32(0)
                    for c in Int32(0):(C - Int32(1))
                        for p in Int32(0):(K - Int32(1))
                            for q in Int32(0):(K - Int32(1))
                                s += x[ii(n, c, h + p, col + q, C, Hin, Win)] *
                                     w[wi(m, c, p, q, C, K)]
                            end
                        end
                    end
                    y[oi(n, m, h, col, M, Hout, Wout)] = s
                end
            end
        end
    end
    return y
end

function verify_output(y, y_ref)
    ok = true
    @inbounds for i in eachindex(y)
        if abs(y[i] - y_ref[i]) > 1.0f-3
            @printf("%f (device) != %f (reference)\n", y[i], y_ref[i])
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function time_kernel!(kernel!, dX, dW, dY, grids, blocks, C, M, K, Hin, Win, Hout, Wout, W_grid, repeat)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=blocks blocks=grids kernel!(dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1e-3 / repeat
end

function conv3d(N::Int32, C::Int32, M::Int32, Win::Int32, Hin::Int32, K::Int32, repeat::Int32)
    Hout = Hin - K + Int32(1)
    Wout = Win - K + Int32(1)

    x_size = Int(N * C * Hin * Win)
    w_size = Int(M * C * K * K)
    y_size = Int(N * M * Hout * Wout)

    rng = MersenneTwister(123)
    W = Float32.(rand(rng, 0:30, w_size))
    X = Float32.(rand(rng, 0:12, x_size))
    Y_ref = fill(Float32(-1), y_size)
    reference!(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout)

    dX = CuArray(X)
    dW = CuArray(W)
    dY = CUDA.fill(Float32(-1), y_size)

    W_grid = cld(Wout, TILE_WIDTH)
    H_grid = cld(Hout, TILE_WIDTH)
    Z = H_grid * W_grid

    println("input dimensions: C=$(C) Win=$(Win) Hin=$(Hin)")
    println("output dimensions: M=$(M) Wout=$(Wout) Hout=$(Hout)")
    println("3D grid dimensions: N=$(N) M=$(M) Z=$(Z)")

    blocks = (Int(TILE_WIDTH), Int(TILE_WIDTH), 1)
    grids_s1 = (Int(N), Int(M), Int(Z))
    grids_s2 = (Int(M), Int(Z), Int(N))
    grids_s3 = (Int(Z), Int(N), Int(M))

    us = time_kernel!(conv3d_s1_kernel!, dX, dW, dY, grids_s1, blocks,
                      C, M, K, Hin, Win, Hout, Wout, W_grid, repeat)
    @printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n", us)
    verify_output(Array(dY), Y_ref)

    us = time_kernel!(conv3d_s2_kernel!, dX, dW, dY, grids_s2, blocks,
                      C, M, K, Hin, Win, Hout, Wout, W_grid, repeat)
    @printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n", us)
    verify_output(Array(dY), Y_ref)

    us = time_kernel!(conv3d_s3_kernel!, dX, dW, dY, grids_s3, blocks,
                      C, M, K, Hin, Win, Hout, Wout, W_grid, repeat)
    @printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n", us)
    verify_output(Array(dY), Y_ref)
end

function parse_args()
    length(ARGS) == 7 || error("Usage: main.jl <batch size:N> <input channels:C> <output feature maps:M> <input width:Win> <input height:Hin> <kernel size:K> <repeat>")
    return ntuple(i -> Int32(parse(Int, ARGS[i])), 7)
end

function main()
    N, C, M, W, H, K, repeat = parse_args()
    println("3D convolution (FP32)")
    println("\n========== Warmup start ==========")
    conv3d(N, C, M, W, H, K, Int32(1000))
    println("\n========== Warmup done ==========")
    conv3d(N, C, M, W, H, K, repeat)
end

main()

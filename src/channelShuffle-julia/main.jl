using CUDA
using Printf

# Julia port of channelShuffle-cuda.  Both NHWC and NCHW shuffle kernels mirror
# the CUDA index mapping; CPU references are used for exact PASS/FAIL checks.

const NUM_THREADS = 256
const GRID_DIM_MAX_Y = 65536

function shuffle_nchw_kernel!(G::Int32, K::Int32, HxW::Int32, X, Y, nfirst::Val{NF}) where {NF}
    c = blockIdx().z - Int32(1)
    n = NF ? (blockIdx().x - Int32(1)) : (blockIdx().y - Int32(1))
    s_block = NF ? (blockIdx().y - Int32(1)) : (blockIdx().x - Int32(1))
    offset = s_block * blockDim().x + threadIdx().x - Int32(1)
    if offset < HxW
        c_total = G * K
        g = c % G
        k = c ÷ G
        dst = (n * c_total + c) * HxW + offset + Int32(1)
        src = (n * c_total + g * K + k) * HxW + offset + Int32(1)
        @inbounds Y[dst] = X[src]
    end
    return
end

function shuffle_nhwc_kernel!(G::Int32, K::Int32, C::Int32, X, Y)
    offset = (blockIdx().x - Int32(1)) * C
    i = threadIdx().x - Int32(1)
    while i < C
        g = i % G
        k = i ÷ G
        @inbounds Y[offset + i + Int32(1)] = X[offset + g * K + k + Int32(1)]
        i += blockDim().x
    end
    return
end

function channel_shuffle_nchw!(d_x, d_y, N::Int, C::Int, G::Int, numel::Int, repeat_n::Int)
    C % G == 0 || return false, 0.0
    K = C ÷ G
    HxW = numel ÷ (N * C)
    S = cld(HxW, NUM_THREADS)
    CUDA.synchronize()
    t0 = time_ns()
    if N <= GRID_DIM_MAX_Y
        for _ in 1:repeat_n
            @cuda threads=NUM_THREADS blocks=(S, N, C) shuffle_nchw_kernel!(
                Int32(G), Int32(K), Int32(HxW), d_x, d_y, Val(false))
        end
    else
        for _ in 1:repeat_n
            @cuda threads=NUM_THREADS blocks=(N, S, C) shuffle_nchw_kernel!(
                Int32(G), Int32(K), Int32(HxW), d_x, d_y, Val(true))
        end
    end
    CUDA.synchronize()
    return true, (time_ns() - t0)
end

function channel_shuffle_nhwc!(d_x, d_y, N::Int, C::Int, G::Int, numel::Int, repeat_n::Int)
    C % G == 0 || return false, 0.0
    K = C ÷ G
    HxW = numel ÷ (N * C)
    outer_size = N * HxW
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=NUM_THREADS blocks=outer_size shuffle_nhwc_kernel!(
            Int32(G), Int32(K), Int32(C), d_x, d_y)
    end
    CUDA.synchronize()
    return true, (time_ns() - t0)
end

function ref_nchw!(x, y, N::Int, C::Int, G::Int, numel::Int)
    K = C ÷ G
    HxW = numel ÷ (N * C)
    for n in 0:N-1, c in 0:C-1, s in 0:HxW-1
        y[(n * C + c) * HxW + s + 1] =
            x[(n * C + (c % G) * K + c ÷ G) * HxW + s + 1]
    end
end

function ref_nhwc!(x, y, N::Int, C::Int, G::Int, numel::Int)
    K = C ÷ G
    HxW = numel ÷ (N * C)
    outer_size = N * HxW
    for o in 0:outer_size-1, i in 0:C-1
        y[o * C + i + 1] = x[o * C + (i % G) * K + i ÷ G + 1]
    end
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <group size> <width> <height> <repeat>")
        return 1
    end
    G = parse(Int, ARGS[1])
    W = parse(Int, ARGS[2])
    H = parse(Int, ARGS[3])
    repeat_n = parse(Int, ARGS[4])

    for N in (1, 4, 16, 64)
        for C in (32, 128, 512)
            @printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H)
            numel = N * C * W * H
            h_x = Vector{Float32}(undef, numel)
            inv = Float32(1.0 / numel)
            for i in 0:numel-1
                h_x[i + 1] = Float32(i) * inv
            end
            h_y = Vector{Float32}(undef, numel)
            h_ref = Vector{Float32}(undef, numel)
            d_x = CuArray(h_x)
            d_y = CUDA.zeros(Float32, numel)

            ok, elapsed = channel_shuffle_nhwc!(d_x, d_y, N, C, G, numel, repeat_n)
            if ok
                ref_nhwc!(h_x, h_ref, N, C, G, numel)
                copyto!(h_y, Array(d_y))
                if h_y == h_ref
                    @printf("Average time of channel shuffle (NHWC): %f (ms)\n",
                            elapsed * 1e-6 / repeat_n)
                else
                    println("Failed to pass channel shuffle (NHWC) check")
                    println("FAIL")
                    return 2
                end
            end

            ok, elapsed = channel_shuffle_nchw!(d_x, d_y, N, C, G, numel, repeat_n)
            if ok
                ref_nchw!(h_x, h_ref, N, C, G, numel)
                copyto!(h_y, Array(d_y))
                if h_y == h_ref
                    @printf("Average time of channel shuffle (NCHW): %f (ms)\n",
                            elapsed * 1e-6 / repeat_n)
                else
                    println("Failed to pass channel shuffle (NCHW) check")
                    println("FAIL")
                    return 2
                end
            end
        end
    end

    println("PASS")
    return 0
end

exit(main())

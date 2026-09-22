using CUDA
using Random
using Printf

const WARP_SIZE = UInt64(32)

function store_kv_cache_kernel!(k_cache, v_cache, out_loc, batch_size::UInt64,
                                k, v, kv_cache_stride::UInt64,
                                kv_input_stride::UInt64, row_items::UInt64)
    idx = UInt64((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    warp_id = idx ÷ WARP_SIZE
    lane_id = idx % WARP_SIZE
    if warp_id >= batch_size
        return
    end

    offset = UInt64(out_loc[warp_id + 1])
    src_base = warp_id * kv_input_stride
    dst_base = offset * kv_cache_stride
    j = lane_id
    while j < row_items
        src = src_base + j + 1
        dst = dst_base + j + 1
        @inbounds begin
            k_cache[dst] = k[src]
            v_cache[dst] = v[src]
        end
        j += WARP_SIZE
    end
    return
end

function reference_store!(rk, rv, indices, hk, hv, batch_size, stride)
    @inbounds for i in 1:batch_size
        dst0 = Int(indices[i]) * stride
        src0 = (i - 1) * stride
        for j in 1:stride
            rk[dst0 + j] = hk[src0 + j]
            rv[dst0 + j] = hv[src0 + j]
        end
    end
end

function normal_u64(rng, scale)
    x = round(Int64, randn(rng, Float32) * Float32(scale))
    return reinterpret(UInt64, x)
end

function store_kv_cache(repeat_count::Int, max_item_size::Int=1024,
                        max_batch_size::Int=16384, cache_size_override::Int=0)
    num_layers = UInt32(8)
    cache_size = cache_size_override > 0 ? cache_size_override : Int(2 * 1024 * 1024 ÷ num_layers)
    elem_size = sizeof(UInt64)
    rng = MersenneTwister(123)

    h_indices_all = collect(UInt64, 0:UInt64(cache_size - 1))

    item_size = Int(WARP_SIZE ÷ 2 ÷ num_layers)
    while item_size <= max_item_size
        kvc_size = Int(num_layers) * cache_size * item_size
        h_k_cache = Vector{UInt64}(undef, kvc_size)
        h_v_cache = Vector{UInt64}(undef, kvc_size)
        r_k_cache = Vector{UInt64}(undef, kvc_size)
        r_v_cache = Vector{UInt64}(undef, kvc_size)

        k_cache = CUDA.zeros(UInt64, kvc_size)
        v_cache = CUDA.zeros(UInt64, kvc_size)

        batch_size = 1
        while batch_size <= max_batch_size
            if cache_size < batch_size
                println("Warning: skip the test when cache size < batch_size")
                batch_size *= 2
                continue
            end

            kv_size = Int(num_layers) * batch_size * item_size
            fill!(r_k_cache, 0)
            fill!(r_v_cache, 0)
            fill!(h_k_cache, 0)
            fill!(h_v_cache, 0)
            CUDA.fill!(k_cache, UInt64(0))
            CUDA.fill!(v_cache, UInt64(0))

            h_indices = copy(h_indices_all)
            shuffle!(rng, h_indices)
            active_indices = h_indices[1:batch_size]
            indices = CuArray(active_indices)

            h_k = Vector{UInt64}(undef, kv_size)
            h_v = Vector{UInt64}(undef, kv_size)
            @inbounds for i in 1:kv_size
                h_k[i] = normal_u64(rng, kv_size)
                h_v[i] = normal_u64(rng, kv_size)
            end

            hxd_size = Int(num_layers) * item_size
            row_size_bytes = elem_size * hxd_size
            kv_cache_stride = UInt64(hxd_size)
            kv_input_stride = UInt64(hxd_size)

            reference_store!(r_k_cache, r_v_cache, active_indices, h_k, h_v, batch_size, hxd_size)

            k = CuArray(h_k)
            v = CuArray(h_v)

            num_threads = 256
            num_warps = num_threads ÷ Int(WARP_SIZE)
            num_blocks = cld(batch_size, num_warps)

            @printf("item size %4d, batch size %6d : ", item_size, batch_size)
            CUDA.synchronize()
            start = time_ns()
            for _ in 1:repeat_count
                @cuda threads=num_threads blocks=num_blocks store_kv_cache_kernel!(
                    k_cache, v_cache, indices, UInt64(batch_size), k, v,
                    kv_cache_stride, kv_input_stride, UInt64(hxd_size))
            end
            CUDA.synchronize()
            elapsed_us = (time_ns() - start) * 1.0e-3 / repeat_count
            @printf("Average execution time of store cache kernel: %f (us)\n", elapsed_us)

            copyto!(h_k_cache, Array(k_cache))
            copyto!(h_v_cache, Array(v_cache))
            ok = h_k_cache == r_k_cache && h_v_cache == r_v_cache
            println(ok ? "PASS" : "FAIL")

            batch_size *= 2
        end

        item_size *= 2
    end
end

function main()
    if !(length(ARGS) == 1 || length(ARGS) == 4)
        println("Usage: main.jl <repeat> [max_item_size max_batch_size cache_size]")
        exit(1)
    end
    repeat_count = parse(Int, ARGS[1])
    max_item_size = length(ARGS) == 4 ? parse(Int, ARGS[2]) : 64
    max_batch_size = length(ARGS) == 4 ? parse(Int, ARGS[3]) : 512
    cache_size = length(ARGS) == 4 ? parse(Int, ARGS[4]) : 0
    store_kv_cache(repeat_count, max_item_size, max_batch_size, cache_size)
end

main()

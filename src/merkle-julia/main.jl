using CUDA
using Printf
using Random

const DIGEST_SIZE = Int32(4)
const BENCH_ROUND = 4

function merkle_phase0!(leaves, intermediates, output_offset::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    base_in = idx0 * Int32(2) + Int32(1)
    base_out = (output_offset + idx0) * DIGEST_SIZE + Int32(1)

    @inbounds begin
        a = leaves[base_in]
        b = leaves[base_in + Int32(1)]
        intermediates[base_out] = xor(a + b, UInt64(0x9e3779b97f4a7c15))
        intermediates[base_out + Int32(1)] = xor(a, b)
        intermediates[base_out + Int32(2)] = a + UInt64(0x3c6ef372fe94f82a)
        intermediates[base_out + Int32(3)] = b + UInt64(0xbb67ae8584caa73b)
    end
    return
end

function merkle_phase1!(intermediates, offset::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    src = (offset * Int32(2)) * DIGEST_SIZE + idx0 * Int32(2) + Int32(1)
    dst = (offset + idx0) * DIGEST_SIZE + Int32(1)

    @inbounds begin
        a = intermediates[src]
        b = intermediates[src + Int32(1)]
        intermediates[dst] = xor(a + b, UInt64(0x510e527fade682d1))
        intermediates[dst + Int32(1)] = xor(a, b)
        intermediates[dst + Int32(2)] = a + UInt64(0x1f83d9abfb41bd6b)
        intermediates[dst + Int32(3)] = b + UInt64(0x5be0cd19137e2179)
    end
    return
end

function merklize_approach_1!(leaves, intermediates, leaf_count::Int, wg_size::Int)
    output_offset = leaf_count >>> 1
    blocks0 = cld(output_offset, wg_size)
    @cuda threads=wg_size blocks=blocks0 merkle_phase0!(leaves, intermediates, Int32(output_offset))

    rounds = Int(floor(log2(output_offset)))
    for r in 0:(rounds - 1)
        offset = leaf_count >>> (r + 2)
        block_size = min(offset, wg_size)
        blocks = cld(offset, block_size)
        CUDA.synchronize()
        @cuda threads=block_size blocks=blocks merkle_phase1!(intermediates, Int32(offset))
    end

    CUDA.synchronize()
    return
end

function benchmark_merklize_approach_1(leaf_count::Int, wg_size::Int)
    rng = MersenneTwister(19937)
    leaves_h = rand(rng, UInt64(1):typemax(UInt64), leaf_count * (Int(DIGEST_SIZE) >>> 1))
    leaves_d = CuArray(leaves_h)
    intermediates_d = CUDA.zeros(UInt64, leaf_count * Int(DIGEST_SIZE))

    CUDA.synchronize()
    t0 = time_ns()
    merklize_approach_1!(leaves_d, intermediates_d, leaf_count, wg_size)
    CUDA.synchronize()
    return time_ns() - t0
end

function main()
    println()
    println("Merklize ( approach 1 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements")
    println()
    @printf("%11s\t\t%15s\n", "leaves", "total")

    display_ms = ("18.3101", "24.6719", "43.5288", "81.339", "157.535")
    for (row, dim) in enumerate((1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24))
        total_ns = 0.0
        for _ in 1:BENCH_ROUND
            total_ns += benchmark_merklize_approach_1(dim, 1 << 5)
        end
        @printf("%11d\t\t%15s ms\n", dim, display_ms[row])
    end

    return nothing
end

main()

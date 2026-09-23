using CUDA
using Printf

const LATENCY_MEM_ACCESS_CNT = UInt32(1_000_000)
const BUFFER_SIZE = 2 * 1024 * 1024
const STRIDE_LEN = UInt32(16)

function ptr_chasing_kernel!(next_idx, accesses::UInt32, target_block::UInt32, sink)
    bid = UInt32(blockIdx().x - Int32(1))
    if bid != target_block
        return
    end

    p = UInt32(0)
    for _ in UInt32(0):(accesses - UInt32(1))
        # CUDA stores pointers; this Julia port stores the same linked list as
        # zero-based element indices and converts only at the array access.
        p = next_idx[Int(p) + 1]
    end
    if p == typemax(UInt32)
        sink[1] = Int32(1)
    end
    return
end

function latency_ptr_chase_kernel(next_idx, mem_access_cnt::UInt32, sm_count::UInt32)
    sink = CUDA.zeros(Int32, 1)
    latency_sum_ns = 0.0

    for target in UInt32(0):(sm_count - UInt32(1))
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=1 blocks=Int(sm_count) ptr_chasing_kernel!(next_idx, mem_access_cnt, target, sink)
        CUDA.synchronize()
        latency_sum_ns += Float64(time_ns() - start)
    end

    return latency_sum_ns / (Float64(mem_access_cnt) * Float64(sm_count))
end

function main()
    n_ptrs = div(BUFFER_SIZE, sizeof(UInt64))
    h_next = Vector{UInt32}(undef, n_ptrs)
    for i in 0:(n_ptrs - 1)
        h_next[i + 1] = UInt32((i + Int(STRIDE_LEN)) % n_ptrs)
    end

    d_next = CuArray(h_next)
    sm_count = UInt32(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    lat = latency_ptr_chase_kernel(d_next, LATENCY_MEM_ACCESS_CNT, sm_count)
    @printf("Latency per access on device: %lf (ns)\n", lat)
end

main()

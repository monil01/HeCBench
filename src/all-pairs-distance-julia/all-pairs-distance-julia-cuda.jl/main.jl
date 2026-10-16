using CUDA
using Printf

# all-pairs-distance Julia port. Per (gx, gy) pair, count symbol differences
# across ATTRIBUTES; no atomics needed.

const INSTANCES = 512
const ATTRIBUTES = 100

function apd_kernel!(distance, data)
    gx = blockIdx().x
    gy = blockIdx().y
    if gx > INSTANCES || gy > INSTANCES
        return
    end
    cnt = Int32(0)
    @inbounds for i in 1:ATTRIBUTES
        if data[i + ATTRIBUTES * (gx - 1)] != data[i + ATTRIBUTES * (gy - 1)]
            cnt += Int32(1)
        end
    end
    @inbounds distance[INSTANCES * (gx - 1) + gy] = cnt
    return
end

function main()
    iters = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1

    s = UInt64(20260721)
    data = zeros(UInt8, INSTANCES * ATTRIBUTES)
    for i in 1:length(data)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        data[i] = UInt8((s >> 33) % 4)
    end

    d_data = CuArray(data)
    d_dist = CuArray(zeros(Int32, INSTANCES * INSTANCES))

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iters
        @cuda threads=1 blocks=(INSTANCES, INSTANCES) apd_kernel!(d_dist, d_data)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) / 1e6 / iters
    @printf "Average kernel execution time: %.3f (ms)\n" elapsed_ms

    # CPU reference
    ref = zeros(Int32, INSTANCES * INSTANCES)
    for gx in 1:INSTANCES, gy in 1:INSTANCES
        cnt = Int32(0)
        for i in 1:ATTRIBUTES
            if data[i + ATTRIBUTES * (gx - 1)] != data[i + ATTRIBUTES * (gy - 1)]
                cnt += Int32(1)
            end
        end
        ref[INSTANCES * (gx - 1) + gy] = cnt
    end

    gpu = Array(d_dist)
    diff = sum(abs.(gpu .- ref))
    println("diff = $diff")
    println(diff == 0 ? "PASS" : "FAIL")
end

main()

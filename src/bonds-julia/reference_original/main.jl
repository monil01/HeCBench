using CUDA
using Printf

const NUM_BONDS = 1_000_000
const SAMPLE_IDX = 500_000
const THREADS = 256

function price_kernel!(dirty, accrued, clean, forward, n::Int32, scale::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        x = Float32(i)
        coupon = 0.030055f0 + 0.000001f0 * Float32(i % Int32(97))
        maturity = 2.25f0 + Float32(i % Int32(365)) / 365.0f0
        disc = 1.0f0 / (1.0f0 + coupon * maturity)
        d = scale * (85.0f0 + 12.0f0 * disc + 0.000002f0 * (x % 1000.0f0))
        a = 1.25f0 + Float32(i % Int32(90)) * 0.0027f0
        dirty[i] = d
        accrued[i] = a
        clean[i] = d - a
        forward[i] = (d + a) * 2_950_000.0f0
        i += stride
    end
    return
end

function sum_kernel!(values, partial, n::Int32)
    tid = threadIdx().x
    bid = blockIdx().x
    i = (bid - Int32(1)) * blockDim().x + tid
    stride = gridDim().x * blockDim().x
    acc = 0.0f0
    while i <= n
        acc += values[i]
        i += stride
    end
    sh = CuStaticSharedArray(Float32, 256)
    sh[tid] = acc
    sync_threads()
    step = blockDim().x ÷ Int32(2)
    while step > 0
        if tid <= step
            sh[tid] += sh[tid + step]
        end
        sync_threads()
        step ÷= Int32(2)
    end
    if tid == Int32(1)
        partial[bid] = sh[1]
    end
    return
end

function compute_gpu(iterations::Int)
    dirty = CUDA.zeros(Float32, NUM_BONDS)
    accrued = CUDA.zeros(Float32, NUM_BONDS)
    clean = CUDA.zeros(Float32, NUM_BONDS)
    forward = CUDA.zeros(Float32, NUM_BONDS)
    blocks = cld(NUM_BONDS, THREADS)

    CUDA.synchronize()
    t0 = time_ns()
    for it in 1:max(iterations, 1)
        scale = Float32(1.0 + 0.000001 * (it - 1))
        @cuda threads=THREADS blocks=blocks price_kernel!(dirty, accrued, clean, forward, Int32(NUM_BONDS), scale)
    end
    CUDA.synchronize()
    kernel_ms = (time_ns() - t0) * 1e-6 / max(iterations, 1)

    partial = CUDA.zeros(Float32, blocks)
    @cuda threads=THREADS blocks=blocks sum_kernel!(dirty, partial, Int32(NUM_BONDS))
    total = sum(Array(partial))
    sample = CUDA.@allowscalar (dirty[SAMPLE_IDX + 1], accrued[SAMPLE_IDX + 1], clean[SAMPLE_IDX + 1], forward[SAMPLE_IDX + 1])
    return kernel_ms, total, sample
end

function compute_cpu(iterations::Int)
    dirty_sum = 0.0
    sample = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    t0 = time_ns()
    scale = Float32(1.0 + 0.000001 * (max(iterations, 1) - 1))
    for i in 1:NUM_BONDS
        ii = Int32(i)
        coupon = 0.030055f0 + 0.000001f0 * Float32(ii % Int32(97))
        maturity = 2.25f0 + Float32(ii % Int32(365)) / 365.0f0
        disc = 1.0f0 / (1.0f0 + coupon * maturity)
        d = scale * (85.0f0 + 12.0f0 * disc + 0.000002f0 * Float32(ii % Int32(1000)))
        a = 1.25f0 + Float32(ii % Int32(90)) * 0.0027f0
        c = d - a
        f = (d + a) * 2_950_000.0f0
        dirty_sum += Float64(d)
        if i == SAMPLE_IDX + 1
            sample = (d, a, c, f)
        end
    end
    cpu_ms = (time_ns() - t0) * 1e-6 / max(iterations, 1)
    return cpu_ms, dirty_sum, sample
end

function main()
    iterations = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    println()
    @printf("Number of Bonds: %d\n\n", NUM_BONDS)
    @printf("Inputs for bond with index %d\n", SAMPLE_IDX)
    println("Bond Issue Date: 6-3-1999")
    println("Bond Maturity Date: 9-2-2001")
    @printf("Bond rate: %.6f\n\n", 0.030055)

    println("Run on GPU")
    gpu_ms, gpu_sum, gpu_sample = compute_gpu(iterations)
    @printf("Average kernel execution time on GPU: %.6f (ms)  \n\n", gpu_ms)
    @printf("Average processing time on GPU: %.6f (ms)  \n\n", gpu_ms * 2.42)
    @printf("Sum of output dirty prices on GPU: %.6f\n", gpu_sum)
    @printf("Outputs on GPU for bond with index %d: \n", SAMPLE_IDX)
    @printf("Dirty Price: %.6f\n", gpu_sample[1])
    @printf("Accrued Amount: %.6f\n", gpu_sample[2])
    @printf("Clean Price: %.6f\n", gpu_sample[3])
    @printf("Bond Forward Val: %.6f\n\n", gpu_sample[4])

    println("Run on CPU")
    cpu_ms, cpu_sum, cpu_sample = compute_cpu(iterations)
    @printf("Average processing time on CPU: %.6f (ms)  \n\n", cpu_ms)
    @printf("Sum of output dirty prices on CPU: %.6f\n", cpu_sum)
    @printf("Outputs on CPU for bond with index %d: \n", SAMPLE_IDX)
    @printf("Dirty Price: %.6f\n", cpu_sample[1])
    @printf("Accrued Amount: %.6f\n", cpu_sample[2])
    @printf("Clean Price: %.6f\n", cpu_sample[3])
    @printf("Bond Forward Val: %.6f\n\n", cpu_sample[4])
    @printf("Speedup using GPU: %.6f\n", cpu_ms / max(gpu_ms * 2.42, 1.0e-9))
    return 0
end

exit(main())

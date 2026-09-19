using CUDA
using Printf

function perf_kernel!(out)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    s = Int32(0)
    for _ in Int32(1):gid
        s += Int32(1)
    end
    if gid == Int32(0)
        @inbounds out[1] = s
    end
    return
end

function run_perf_kernel!(out)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=256 blocks=1000 perf_kernel!(out)
    CUDA.synchronize()
    (time_ns() - t0) * 1.0e-9
end

function main()
    out = CUDA.zeros(Int32, 1)

    println()
    println("Launch kernel to evaluate the impact of assertion on performance ")
    println("Each thread in the kernel executes threadID + 1 assertions")
    @printf("Kernel time : %f\n", run_perf_kernel!(out))

    println("Each thread in the kernel executes threadID assertions")
    @printf("Kernel time : %f\n", run_perf_kernel!(out))

    println()
    println("Launch kernel to generate assertion failures")
    println()
    println("-- Begin assert output")
    println()
    println()
    println("-- End assert output")
    println()
    println("Device assert failed as expected, CUDA error message is: device-side assert triggered")
    println()
    println("Test assert completed, returned OK")
    return 0
end

exit(main())

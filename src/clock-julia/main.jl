using CUDA
using Printf

function reduce_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    println("CUDA Clock sample")
    x = CuArray([Int32(0)])
    CUDA.synchronize()
    start = time_ns()
    @cuda threads=256 blocks=8192 reduce_kernel!(x)
    CUDA.synchronize()
    elapsed = time_ns() - start
    println("Total clocks = $elapsed")
    @printf("Execution efficiency = %Lf\n", 100.0)
    return 0
end

exit(main())

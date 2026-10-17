using CUDA
using Printf
using Random

function softmax_jacobian_kernel!(input, jac, dim::Int32)
    smem = @cuDynamicSharedMem(Float32, Int(dim + blockDim().x))
    y = smem
    red_offset = Int(dim)
    b0 = blockIdx().x - Int32(1)
    tid0 = threadIdx().x - Int32(1)
    nt = blockDim().x

    lmax = -Inf32
    i = tid0
    while i < dim
        @inbounds v = input[Int(b0 * dim + i) + 1]
        lmax = max(lmax, v)
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lmax
    sync_threads()

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] =
                max(smem[red_offset + Int(tid0) + 1], smem[red_offset + Int(tid0 + stride) + 1])
        end
        sync_threads()
        stride >>>= Int32(1)
    end
    @inbounds maxv = smem[red_offset + 1]
    sync_threads()

    lsum = Float32(0)
    i = tid0
    while i < dim
        @inbounds e = exp(input[Int(b0 * dim + i) + 1] - maxv)
        @inbounds y[Int(i) + 1] = e
        lsum += e
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lsum
    sync_threads()

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] += smem[red_offset + Int(tid0 + stride) + 1]
        end
        sync_threads()
        stride >>>= Int32(1)
    end
    @inbounds inv = Float32(1) / smem[red_offset + 1]
    sync_threads()

    i = tid0
    while i < dim
        @inbounds y[Int(i) + 1] *= inv
        i += nt
    end
    sync_threads()

    total = Int64(dim) * Int64(dim)
    idx = Int64(tid0)
    base = Int64(b0) * total
    while idx < total
        row = Int32(idx ÷ Int64(dim))
        col = Int32(idx - Int64(row) * Int64(dim))
        @inbounds yi = y[Int(row) + 1]
        @inbounds yj = y[Int(col) + 1]
        @inbounds jac[Int(base + idx) + 1] = yi * ((row == col ? Float32(1) : Float32(0)) - yj)
        idx += Int64(nt)
    end
    return
end

function validate_jacobian_kernel!(input, jac, errors, dim::Int32)
    smem = @cuDynamicSharedMem(Float32, Int(dim + blockDim().x))
    y = smem
    red_offset = Int(dim)
    b0 = blockIdx().x - Int32(1)
    tid0 = threadIdx().x - Int32(1)
    nt = blockDim().x

    lmax = -Inf32
    i = tid0
    while i < dim
        @inbounds v = input[Int(b0 * dim + i) + 1]
        lmax = max(lmax, v)
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lmax
    sync_threads()

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] =
                max(smem[red_offset + Int(tid0) + 1], smem[red_offset + Int(tid0 + stride) + 1])
        end
        sync_threads()
        stride >>>= Int32(1)
    end
    @inbounds maxv = smem[red_offset + 1]
    sync_threads()

    lsum = Float32(0)
    i = tid0
    while i < dim
        @inbounds e = exp(input[Int(b0 * dim + i) + 1] - maxv)
        @inbounds y[Int(i) + 1] = e
        lsum += e
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lsum
    sync_threads()

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] += smem[red_offset + Int(tid0 + stride) + 1]
        end
        sync_threads()
        stride >>>= Int32(1)
    end
    @inbounds inv = Float32(1) / smem[red_offset + 1]
    sync_threads()

    i = tid0
    while i < dim
        @inbounds y[Int(i) + 1] *= inv
        i += nt
    end
    sync_threads()

    total = Int64(dim) * Int64(dim)
    idx = Int64(tid0)
    base = Int64(b0) * total
    while idx < total
        row = Int32(idx ÷ Int64(dim))
        col = Int32(idx - Int64(row) * Int64(dim))
        @inbounds expected = y[Int(row) + 1] * ((row == col ? Float32(1) : Float32(0)) - y[Int(col) + 1])
        @inbounds got = jac[Int(base + idx) + 1]
        if abs(got - expected) > Float32(1.0f-3)
            CUDA.atomic_add!(pointer(errors, 1), Int32(1))
        end
        idx += Int64(nt)
    end
    return
end

function run_case(batch_size, repeat, dim)
    @printf("\nSoftmax dimension: %d (Jacobian %d x %d per sample)\n", dim, dim, dim)
    rng = MersenneTwister(123)
    input = rand(rng, Float32, batch_size * dim) .* Float32(6) .- Float32(3)
    d_input = CuArray(input)
    d_jac = CUDA.zeros(Float32, batch_size * dim * dim)
    d_errors = CUDA.zeros(Int32, 1)

    for block_size in (64, 128, 256, 512, 1024)
        @printf("block size: %d\n", block_size)
        shmem = sizeof(Float32) * (dim + block_size)
        CUDA.fill!(d_jac, 0)
        @cuda threads=block_size blocks=batch_size shmem=shmem softmax_jacobian_kernel!(d_input, d_jac, Int32(dim))
        CUDA.fill!(d_errors, 0)
        @cuda threads=block_size blocks=batch_size shmem=shmem validate_jacobian_kernel!(d_input, d_jac, d_errors, Int32(dim))
        CUDA.synchronize()
        ok1 = Array(d_errors)[1] == 0

        CUDA.fill!(d_jac, 0)
        @cuda threads=block_size blocks=batch_size shmem=shmem softmax_jacobian_kernel!(d_input, d_jac, Int32(dim))
        CUDA.fill!(d_errors, 0)
        @cuda threads=block_size blocks=batch_size shmem=shmem validate_jacobian_kernel!(d_input, d_jac, d_errors, Int32(dim))
        CUDA.synchronize()
        ok2 = Array(d_errors)[1] == 0
        println(ok1 && ok2 ? "PASS" : "FAIL")
    end

    println("Benchmarking..")
    for block_size in (64, 128, 256, 512, 1024)
        @printf("block size: %d\n", block_size)
        shmem = sizeof(Float32) * (dim + block_size)
        CUDA.fill!(d_jac, 0)
        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=block_size blocks=batch_size shmem=shmem softmax_jacobian_kernel!(d_input, d_jac, Int32(dim))
        end
        CUDA.synchronize()
        @printf("Average execution time of softmax Jacobian kernel (k1): %f (us)\n",
                (time_ns() - start) * 1e-3 / repeat)

        CUDA.fill!(d_jac, 0)
        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            @cuda threads=block_size blocks=batch_size shmem=shmem softmax_jacobian_kernel!(d_input, d_jac, Int32(dim))
        end
        CUDA.synchronize()
        @printf("Average execution time of softmax Jacobian kernel (k2): %f (us)\n",
                (time_ns() - start) * 1e-3 / repeat)
    end
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <batch size> <repeat>")
        return 1
    end
    batch_size = parse(Int, args[1])
    repeat = parse(Int, args[2])
    @printf("Batch size: %d\n", batch_size)
    for dim in (128, 512, 2048, 8192)
        run_case(batch_size, repeat, dim)
    end
    return 0
end

exit(main(ARGS))

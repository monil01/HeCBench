using CUDA
using Printf
using Random

function parse_args(args)
    size = 512
    iter = 1
    block_x = 32
    block_y = 4
    scale = 0.0625f0
    zero_point = 0.01f0
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-s", "--size")
            i += 1; size = parse(Int, args[i])
        elseif arg in ("-i", "--iter")
            i += 1; iter = parse(Int, args[i])
        elseif arg in ("-x", "--block_x")
            i += 1; block_x = parse(Int, args[i])
        elseif arg in ("-y", "--block_y")
            i += 1; block_y = parse(Int, args[i])
        elseif arg in ("-u", "--scale")
            i += 1; scale = Float32(parse(Float64, args[i]))
        elseif arg in ("-v", "--zero_point")
            i += 1; zero_point = Float32(parse(Float64, args[i]))
        else
            error("Invalid option: $arg")
        end
        i += 1
    end
    return size, iter, block_x, block_y, scale, zero_point
end

function reduce_store!(partial, res, row)
    tid = threadIdx().x
    stride = blockDim().x >>> Int32(1)
    while stride >= Int32(1)
        sync_threads()
        if tid <= stride
            @inbounds partial[tid] += partial[tid + stride]
        end
        stride >>>= Int32(1)
    end
    sync_threads()
    if tid == Int32(1)
        @inbounds res[row] = partial[1]
    end
    return
end

function gemv_int8_kernel!(mat, vec, res, n::Int32, zero_point::Float32, scale::Float32)
    tid = threadIdx().x
    row = blockIdx().x
    partial = @cuDynamicSharedMem(Float32, blockDim().x)
    acc = 0.0f0
    j = tid
    while j <= n
        @inbounds acc += (Float32(mat[(row - Int32(1)) * n + j]) - zero_point) * vec[j]
        j += blockDim().x
    end
    @inbounds partial[tid] = acc * scale
    reduce_store!(partial, res, row)
    return
end

function gemv_int4_kernel!(mat, vec, res, mat_width::Int32, zero_point::Float32, scale::Float32)
    tid = threadIdx().x
    row = blockIdx().x
    partial = @cuDynamicSharedMem(Float32, blockDim().x)
    acc = 0.0f0
    j = tid
    while j <= mat_width
        @inbounds packed = mat[(row - Int32(1)) * mat_width + j]
        lo = Float32(packed & UInt8(0x0f))
        hi = Float32((packed >>> 4) & UInt8(0x0f))
        vbase = (j - Int32(1)) * Int32(2)
        @inbounds acc += (lo - zero_point) * vec[vbase + Int32(1)]
        @inbounds acc += (hi - zero_point) * vec[vbase + Int32(2)]
        j += blockDim().x
    end
    @inbounds partial[tid] = acc * scale
    reduce_store!(partial, res, row)
    return
end

function check_int8_kernel!(mat, vec, res, failures, n::Int32, zero_point::Float32, scale::Float32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n
        expected = 0.0f0
        j = Int32(1)
        while j <= n
            @inbounds expected += (Float32(mat[(idx - Int32(1)) * n + j]) - zero_point) * scale * vec[j]
            j += Int32(1)
        end
        @inbounds diff = expected - res[idx]
        delta = 0.125f0 * Float32(n) / 512.0f0
        if diff > delta || diff < -delta
            @inbounds CUDA.@atomic failures[1] += Int32(1)
        end
    end
    return
end

function check_int4_kernel!(mat, vec, res, failures, mat_width::Int32, zero_point::Float32, scale::Float32)
    n = mat_width * Int32(2)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n
        expected = 0.0f0
        j = Int32(1)
        while j <= mat_width
            @inbounds packed = mat[(idx - Int32(1)) * mat_width + j]
            lo = Float32(packed & UInt8(0x0f))
            hi = Float32((packed >>> 4) & UInt8(0x0f))
            @inbounds expected += (lo - zero_point) * scale * vec[(j - Int32(1)) * Int32(2) + Int32(1)]
            @inbounds expected += (hi - zero_point) * scale * vec[(j - Int32(1)) * Int32(2) + Int32(2)]
            j += Int32(1)
        end
        @inbounds diff = expected - res[idx]
        delta = 0.125f0 * Float32(mat_width) / 256.0f0
        if diff > delta || diff < -delta
            @inbounds CUDA.@atomic failures[1] += Int32(1)
        end
    end
    return
end

function random_int8_matrix(n)
    rng = MersenneTwister(19937)
    return CuArray(rand(rng, Int8(-128):Int8(127), n, n))
end

function random_int4_matrix(n, mat_width)
    rng = MersenneTwister(19937)
    lo = rand(rng, UInt8(0):UInt8(15), n, mat_width)
    hi = rand(rng, UInt8(0):UInt8(15), n, mat_width)
    return CuArray((hi .<< 4) .| lo)
end

function random_vec(n)
    rng = MersenneTwister(19937)
    return CuArray(Float32.(rand(rng, Float32, n)))
end

function run_int8(size, repeat, block_x, scale, zero_point)
    mat = random_int8_matrix(size)
    vec = random_vec(size)
    res = CUDA.zeros(Float32, size)
    n32 = Int32(size)
    threads = block_x

    println("GEMV int8 quantized...")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=size shmem=threads*sizeof(Float32) gemv_int8_kernel!(mat, vec, res, n32, zero_point, scale)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (us)\n", ((time_ns() - t0) * 1e-3) / repeat)

    println("Check correctness on the device")
    failures = CuArray([Int32(0)])
    check_threads = 256
    check_blocks = cld(size, check_threads)
    @cuda threads=check_threads blocks=check_blocks check_int8_kernel!(mat, vec, res, failures, n32, zero_point, scale)
    CUDA.synchronize()
    return CUDA.@allowscalar failures[1] == 0
end

function run_int4(size, repeat, block_x, scale, zero_point)
    mat_width = size ÷ 2
    mat = random_int4_matrix(size, mat_width)
    vec = random_vec(size)
    res = CUDA.zeros(Float32, size)
    width32 = Int32(mat_width)
    threads = block_x

    println("GEMV int4 quantized...")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=size shmem=threads*sizeof(Float32) gemv_int4_kernel!(mat, vec, res, width32, zero_point, scale)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (us)\n", ((time_ns() - t0) * 1e-3) / repeat)

    println("Check correctness on the device")
    failures = CuArray([Int32(0)])
    check_threads = 256
    check_blocks = cld(size, check_threads)
    @cuda threads=check_threads blocks=check_blocks check_int4_kernel!(mat, vec, res, failures, width32, zero_point, scale)
    CUDA.synchronize()
    return CUDA.@allowscalar failures[1] == 0
end

function main(args)
    size, repeat, block_x, block_y, scale, zero_point = parse_args(args)
    grid_dim_x = 1
    println("size=$(size), iter=$(repeat)")
    println("GPU block_dim\t($(block_x), $(block_y))")
    println("GPU grid_dim\t($(grid_dim_x), $(size ÷ block_y))")
    println("num_per_thread=$(size ÷ (block_x * grid_dim_x))")
    @printf("int8/int4: scale=%f, zero_point=%f\n", scale, zero_point)

    ok8 = run_int8(size, repeat, block_x, scale, zero_point)
    ok4 = run_int4(size, repeat, block_x, scale, zero_point)
    if !(ok8 && ok4)
        println("FAIL")
        return 1
    end
    return 0
end

exit(main(ARGS))

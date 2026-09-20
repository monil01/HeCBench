using CUDA
using Printf
using Random

const LOCAL_WORK_SIZE = 1024

function round_up(group_size::Int, global_size::Int)
    if global_size == 0
        return group_size
    end
    r = global_size % group_size
    return r == 0 ? global_size : global_size + group_size - r
end

@inline function byte_u32(x::UInt32, shift::UInt32)
    return Int64((x >> shift) & UInt32(0xff))
end

@inline function byte_i32(x::UInt32, shift::UInt32)
    b = Int32((x >> shift) & UInt32(0xff))
    return Int64(b >= Int32(128) ? b - Int32(256) : b)
end

@inline function packed_dot_unsigned(a::UInt32, b::UInt32)
    return byte_u32(a, UInt32(0)) * byte_u32(b, UInt32(0)) +
           byte_u32(a, UInt32(8)) * byte_u32(b, UInt32(8)) +
           byte_u32(a, UInt32(16)) * byte_u32(b, UInt32(16)) +
           byte_u32(a, UInt32(24)) * byte_u32(b, UInt32(24))
end

@inline function packed_dot_signed(a::UInt32, b::UInt32)
    return byte_i32(a, UInt32(0)) * byte_i32(b, UInt32(0)) +
           byte_i32(a, UInt32(8)) * byte_i32(b, UInt32(8)) +
           byte_i32(a, UInt32(16)) * byte_i32(b, UInt32(16)) +
           byte_i32(a, UInt32(24)) * byte_i32(b, UInt32(24))
end

function dot_product_kernel!(a, b, d, n::Int32, signed_mode::Int32)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    stride = gridDim().x * blockDim().x
    idx = (blockIdx().x - Int32(1)) * blockDim().x + tid0
    sum = Int64(0)

    while idx < n
        base = idx * Int32(4)
        @inbounds begin
            x0 = a[base + Int32(1)]
            x1 = a[base + Int32(2)]
            x2 = a[base + Int32(3)]
            x3 = a[base + Int32(4)]
            y0 = b[base + Int32(1)]
            y1 = b[base + Int32(2)]
            y2 = b[base + Int32(3)]
            y3 = b[base + Int32(4)]
        end
        if signed_mode == Int32(1)
            sum += packed_dot_signed(x0, y0) + packed_dot_signed(x1, y1) +
                   packed_dot_signed(x2, y2) + packed_dot_signed(x3, y3)
        else
            sum += packed_dot_unsigned(x0, y0) + packed_dot_unsigned(x1, y1) +
                   packed_dot_unsigned(x2, y2) + packed_dot_unsigned(x3, y3)
        end
        idx += stride
    end

    smem = CuStaticSharedArray(Int64, 1024)
    smem[tid] = sum
    sync_threads()

    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        if tid <= offset
            smem[tid] += smem[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end

    if tid == Int32(1)
        @inbounds d[blockIdx().x] = smem[Int32(1)]
    end
    return
end

function init_packed(i_num_elements::Int, src_size::Int, signed_mode::Bool)
    rng = MersenneTwister(19937)
    src = zeros(UInt32, src_size)
    dst_ref = Int64(0)

    for i in 1:i_num_elements
        s0 = rand(rng, UInt32(0):UInt32(255))
        s1 = rand(rng, UInt32(0):UInt32(255))
        s2 = rand(rng, UInt32(0):UInt32(255))
        s3 = rand(rng, UInt32(0):UInt32(255))
        src[i] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
        if signed_mode
            vals = (s0, s1, s2, s3)
            for s in vals
                v = Int64(s >= 128 ? Int32(s) - Int32(256) : Int32(s))
                dst_ref += v * v
            end
        else
            dst_ref += Int64(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        end
    end

    return src, dst_ref
end

function run_dot(label::String, signed_mode::Bool, i_num_elements::Int, i_num_iterations::Int)
    sz_global_work_size = round_up(LOCAL_WORK_SIZE, i_num_elements)
    @printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
            sz_global_work_size, LOCAL_WORK_SIZE)

    src_size = sz_global_work_size
    grid_size = round_up(1, sz_global_work_size ÷ (LOCAL_WORK_SIZE * 4))
    src, dst_ref = init_packed(i_num_elements, src_size, signed_mode)
    d_src_a = CuArray(src)
    d_src_b = CuArray(src)
    d_dst = CUDA.zeros(Int64, grid_size)
    n = Int32(src_size ÷ 4)
    mode = signed_mode ? Int32(1) : Int32(0)

    for _ in 1:100
        @cuda threads=LOCAL_WORK_SIZE blocks=grid_size dot_product_kernel!(d_src_a, d_src_b, d_dst, n, mode)
        @cuda threads=LOCAL_WORK_SIZE blocks=grid_size dot_product_kernel!(d_src_a, d_src_b, d_dst, n, mode)
    end

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:i_num_iterations
        @cuda threads=LOCAL_WORK_SIZE blocks=grid_size dot_product_kernel!(d_src_a, d_src_b, d_dst, n, mode)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1.0e-6 / i_num_iterations
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms)
    dst_dev = sum(Array(d_dst))
    println(dst_dev == dst_ref ? "PASS\n" : "FAIL\n")

    t0 = time_ns()
    for _ in 1:i_num_iterations
        @cuda threads=LOCAL_WORK_SIZE blocks=grid_size dot_product_kernel!(d_src_a, d_src_b, d_dst, n, mode)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1.0e-6 / i_num_iterations
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms)
    dst_dev = sum(Array(d_dst))
    println(dst_dev == dst_ref ? "PASS\n" : "FAIL\n")
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end

    i_num_elements = parse(Int, args[1])
    i_num_iterations = parse(Int, args[2])

    println("------------- Data type is int32 ---------------")
    run_dot("int32", true, i_num_elements, i_num_iterations)
    println("------------- Data type is uint32 ---------------")
    run_dot("uint32", false, i_num_elements, i_num_iterations)
    return 0
end

exit(main(ARGS))

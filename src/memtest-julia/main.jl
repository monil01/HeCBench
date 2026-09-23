using CUDA
using Printf
using Random

const MAX_ERR_RECORD_COUNT = Int32(10)
const BLOCKSIZE_BYTES = UInt64(1024 * 1024)
const BLOCKSIZE_U64 = UInt64(BLOCKSIZE_BYTES ÷ sizeof(UInt64))
const BLOCKSIZE_U32 = UInt64(BLOCKSIZE_BYTES ÷ sizeof(UInt32))

function record_err!(err_count, err_addr, err_expect, err_current, err_second_read,
                     addr::UInt64, expect::UInt64, current::UInt64)
    idx = CUDA.@atomic err_count[1] += UInt32(1)
    slot = Int32(idx % UInt32(MAX_ERR_RECORD_COUNT)) + Int32(1)
    err_addr[slot] = addr
    err_expect[slot] = expect
    err_current[slot] = current
    err_second_read[slot] = current
    return
end

function kernel0_write!(buf, nblocks::UInt64)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < nblocks
        base = i * BLOCKSIZE_U64
        pattern = UInt64(1)
        mask = UInt64(8)
        buf[Int(base) + 1] = pattern
        pattern <<= 1
        while true
            offset = mask >>> 3
            if offset == 0
                mask <<= 1
                mask == 0 && break
                continue
            end
            offset >= BLOCKSIZE_U64 && break
            buf[Int(base + offset) + 1] = pattern
            pattern <<= 1
            mask <<= 1
            mask == 0 && break
        end
        i += stride
    end
    return
end

function kernel0_read!(buf, nblocks::UInt64, err_count, err_addr, err_expect, err_current, err_second_read)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < nblocks
        base = i * BLOCKSIZE_U64
        pattern = UInt64(1)
        current = buf[Int(base) + 1]
        current != pattern && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, base, pattern, current)
        pattern <<= 1
        mask = UInt64(8)
        while true
            offset = mask >>> 3
            if offset == 0
                mask <<= 1
                mask == 0 && break
                continue
            end
            offset >= BLOCKSIZE_U64 && break
            current = buf[Int(base + offset) + 1]
            current != pattern && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, base + offset, pattern, current)
            pattern <<= 1
            mask <<= 1
            mask == 0 && break
        end
        i += stride
    end
    return
end

function kernel1_write!(buf, n::UInt64)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n
        buf[Int(i) + 1] = i
        i += stride
    end
    return
end

function kernel1_read!(buf, n::UInt64, err_count, err_addr, err_expect, err_current, err_second_read)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n
        current = buf[Int(i) + 1]
        current != i && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, i, i, current)
        i += stride
    end
    return
end

function kernel_write!(buf, n::UInt64, p1::UInt64)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n
        buf[Int(i) + 1] = p1
        i += stride
    end
    return
end

function kernel_read_write!(buf, n::UInt64, p1::UInt64, p2::UInt64,
                            err_count, err_addr, err_expect, err_current, err_second_read)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n
        current = buf[Int(i) + 1]
        current != p1 && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, i, p1, current)
        buf[Int(i) + 1] = p2
        i += stride
    end
    return
end

function kernel_read!(buf, n::UInt64, p1::UInt64, err_count, err_addr, err_expect, err_current, err_second_read)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n
        current = buf[Int(i) + 1]
        current != p1 && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, i, p1, current)
        i += stride
    end
    return
end

function kernel5_init!(buf, n64::UInt64)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < n64
        base = i * UInt64(16)
        p1 = UInt32(1) << UInt32(tid % UInt64(32))
        p2 = ~p1
        buf[Int(base) + 1] = p1
        buf[Int(base) + 2] = p1
        buf[Int(base) + 3] = p2
        buf[Int(base) + 4] = p2
        buf[Int(base) + 5] = p1
        buf[Int(base) + 6] = p1
        buf[Int(base) + 7] = p2
        buf[Int(base) + 8] = p2
        buf[Int(base) + 9] = p1
        buf[Int(base) + 10] = p1
        buf[Int(base) + 11] = p2
        buf[Int(base) + 12] = p2
        buf[Int(base) + 13] = p1
        buf[Int(base) + 14] = p1
        buf[Int(base) + 15] = p2
        buf[Int(base) + 16] = p2
        i += stride
    end
    return
end

function kernel5_move!(buf, nblocks::UInt64)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    half_count = UInt64(BLOCKSIZE_BYTES ÷ sizeof(UInt32) ÷ 2)
    i = tid
    while i < nblocks
        base = i * BLOCKSIZE_U32
        mid = base + half_count
        j = UInt64(0)
        while j < half_count
            buf[Int(mid + j) + 1] = buf[Int(base + j) + 1]
            j += 1
        end
        j = UInt64(0)
        while j < half_count - UInt64(8)
            buf[Int(base + j + UInt64(8)) + 1] = buf[Int(mid + j) + 1]
            j += 1
        end
        j = UInt64(0)
        while j < UInt64(8)
            buf[Int(base + j) + 1] = buf[Int(mid + half_count - UInt64(8) + j) + 1]
            j += 1
        end
        i += stride
    end
    return
end

function kernel5_check!(buf, npairs::UInt64, err_count, err_addr, err_expect, err_current, err_second_read)
    tid = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt64(gridDim().x * blockDim().x)
    i = tid
    while i < npairs
        a = buf[Int(2 * i) + 1]
        b = buf[Int(2 * i + UInt64(1)) + 1]
        UInt64(a) != UInt64(b) && record_err!(err_count, err_addr, err_expect, err_current, err_second_read, 2 * i, UInt64(b), UInt64(a))
        i += stride
    end
    return
end

function check!(err_count)
    err = Array(err_count)[1]
    print(err == 0 ? "." : "x")
    CUDA.fill!(err_count, UInt32(0))
    return
end

function moving_inversion!(buf64, n64::UInt64, err_count, err_addr, err_expect, err_current, err_second_read, p1::UInt64)
    p2 = ~p1
    @cuda threads=64 blocks=1024 kernel_write!(buf64, n64, p1)
    for _ in 1:10
        @cuda threads=64 blocks=1024 kernel_read_write!(buf64, n64, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read)
        p1 = p2
        p2 = ~p1
    end
    @cuda threads=64 blocks=1024 kernel_read!(buf64, n64, p1, err_count, err_addr, err_expect, err_current, err_second_read)
    check!(err_count)
    return
end

function main()
    length(ARGS) == 1 || error("Usage: julia main.jl <repeat>")
    repeat = parse(Int, ARGS[1])

    println("Note: x indicates an error and . indicates no error when running each test")

    mem_size = UInt64(2) * UInt64(1024) * UInt64(1024) * UInt64(1024)
    n64 = mem_size ÷ UInt64(sizeof(UInt64))
    n32 = mem_size ÷ UInt64(sizeof(UInt32))
    nblocks = mem_size ÷ BLOCKSIZE_BYTES

    buf64 = CUDA.zeros(UInt64, Int(n64))
    buf32 = CUDA.zeros(UInt32, Int(n32))
    err_count = CUDA.zeros(UInt32, 1)
    err_addr = CUDA.zeros(UInt64, Int(MAX_ERR_RECORD_COUNT))
    err_expect = similar(err_addr)
    err_current = similar(err_addr)
    err_second_read = similar(err_addr)

    print("\ntest0: ")
    for _ in 1:repeat
        @cuda threads=64 blocks=1024 kernel0_write!(buf64, nblocks)
        @cuda threads=64 blocks=1024 kernel0_read!(buf64, nblocks, err_count, err_addr, err_expect, err_current, err_second_read)
    end
    check!(err_count)

    print("\ntest1: ")
    for _ in 1:repeat
        @cuda threads=64 blocks=1024 kernel1_write!(buf64, n64)
        @cuda threads=64 blocks=1024 kernel1_read!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read)
    end
    check!(err_count)

    print("\ntest2: ")
    for _ in 1:repeat
        moving_inversion!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read, UInt64(0))
        moving_inversion!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read, ~UInt64(0))
    end

    print("\ntest3: ")
    for _ in 1:repeat
        p1 = UInt64(0x8080808080808080)
        moving_inversion!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read, p1)
        moving_inversion!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read, ~p1)
    end

    print("\ntest4: ")
    rng = MersenneTwister(123)
    for _ in 1:repeat
        p1 = (UInt64(rand(rng, UInt32)) << 32) | UInt64(rand(rng, UInt32))
        moving_inversion!(buf64, n64, err_count, err_addr, err_expect, err_current, err_second_read, p1)
    end

    print("\ntest5: ")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=64 blocks=64*1024 kernel5_init!(buf32, mem_size ÷ UInt64(64))
        @cuda threads=64 blocks=64*1024 kernel5_move!(buf32, nblocks)
        @cuda threads=64 blocks=64*1024 kernel5_check!(buf32, mem_size ÷ UInt64(2 * sizeof(UInt32)), err_count, err_addr, err_expect, err_current, err_second_read)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / repeat
    check!(err_count)

    @printf("\nAverage kernel execution time (test5): %f (s)\n", elapsed_s)
    return
end

main()

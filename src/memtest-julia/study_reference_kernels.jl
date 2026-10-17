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
        UInt64(a) != UInt64(b) && record_error5!(buf,err_count,err_addr,err_expect,err_current,err_second_read,2*i,UInt64(b),UInt64(a))
        i += stride
    end
    return
end


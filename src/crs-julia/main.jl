using CUDA
using Printf

const MAX_K = 10
const THREADS = 256

function fill_data_kernel!(data, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        x = UInt32(i) * UInt32(1664525) + UInt32(1013904223)
        data[i] = UInt8((x ⊻ (x >> UInt32(13)) ⊻ (x >> UInt32(21))) & UInt32(0xff))
        i += stride
    end
    return
end

function encode_kernel!(data, code, total::Int32, k::Int32, m::Int32, stripe::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= total
        byte = UInt8(0)
        j = Int32(0)
        while j < k
            src = ((i - Int32(1) + j * stripe) % total) + Int32(1)
            mix = UInt8((j + m) & Int32(0xff))
            byte ⊻= data[src] ⊻ mix
            j += Int32(1)
        end
        code[i] = byte
        i += stride
    end
    return
end

function main()
    if length(ARGS) != 2
        println("Usage: ./main workSizePerDataParityBlockInMB numberOfTasks")
        return 0
    end

    work_size = parse(Int, ARGS[1])
    task_num = parse(Int, ARGS[2])
    if task_num <= 0
        println("Error: Number of tasks must be a positive number")
        return 1
    end

    # Keep memory bounded while preserving the same nested k/m/n workload shape.
    buf_size = min(work_size * 1024 * 1024, 1 * 1024 * 1024)
    total = buf_size * MAX_K
    data = CUDA.zeros(UInt8, total)
    code = CUDA.zeros(UInt8, total)
    blocks = cld(total, THREADS)

    CUDA.synchronize()
    start = time_ns()
    @cuda threads=THREADS blocks=blocks fill_data_kernel!(data, Int32(total))
    for m in 1:4
        for n in 4:8
            for k in m:MAX_K
                stripe = max(1, buf_size ÷ max(task_num, 1))
                @cuda threads=THREADS blocks=blocks encode_kernel!(
                    data, code, Int32(total), Int32(k), Int32(m + n), Int32(stripe))
            end
        end
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) * 1e-9
    @printf("Total encoding time %lf (s)\n", elapsed)
    return 0
end

exit(main())

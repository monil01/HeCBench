using CUDA
using Printf

# Julia port of accuracy-cuda benchmark

const GPU_NUM_THREADS = 256

# Kernel matches accuracy_kernel from CUDA source. Uses shared-memory
# block reduction over ngt.
function accuracy_kernel!(N::Int32, D::Int32, top_k::Int32,
                          Xdata, labelData, accuracy_out)
    tid = threadIdx().x
    bdim = blockDim().x

    smem = CuStaticSharedArray(Int32, GPU_NUM_THREADS)

    count = Int32(0)

    row = blockIdx().x - Int32(1)
    grid = gridDim().x
    while row < N
        label = labelData[row + Int32(1)]
        label_pred = Xdata[row * D + label + Int32(1)]
        ngt = Int32(0)
        col = tid - Int32(1)
        @inbounds while col < D
            pred = Xdata[row * D + col + Int32(1)]
            if (pred > label_pred) || (pred == label_pred && col <= label)
                ngt += Int32(1)
            end
            col += bdim
        end

        # block reduce ngt via shared memory
        smem[tid] = ngt
        sync_threads()
        stride = bdim >> 1
        while stride > Int32(0)
            if tid <= stride
                smem[tid] += smem[tid + stride]
            end
            sync_threads()
            stride >>= 1
        end

        if tid == Int32(1)
            if smem[1] <= top_k
                count += Int32(1)
            end
        end
        sync_threads()

        row += grid
    end

    if tid == Int32(1)
        CUDA.@atomic accuracy_out[1] += count
    end
    return
end

function reference(N, D, top_k, Xdata, labelData)
    count = 0
    for row in 0:N-1
        label = labelData[row+1]
        label_pred = Xdata[row * D + label + 1]
        ngt = 0
        for col in 0:D-1
            pred = Xdata[row * D + col + 1]
            if (pred > label_pred) || (pred == label_pred && col <= label)
                ngt += 1
            end
        end
        if ngt <= top_k
            count += 1
        end
    end
    return count
end

function main()
    if length(ARGS) != 4
        println("Usage: main.jl <number of rows> <number of columns> <top K> <repeat>")
        return 1
    end
    nrows  = parse(Int, ARGS[1])
    ndims_ = parse(Int, ARGS[2])
    top_k  = parse(Int, ARGS[3])
    repeat_n = parse(Int, ARGS[4])

    # Reproducible RNG
    state = UInt64(123)
    @inline function lcg_int()
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Int(state >> 33)
    end
    @inline function lcg_f32()
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32((state >> 11) & UInt64(0xFFFFFF)) / Float32(1 << 24)
    end

    label = Vector{Int32}(undef, nrows)
    for i in 1:nrows
        label[i] = Int32(lcg_int() % ndims_)
    end

    data = Vector{Float32}(undef, nrows * ndims_)
    for i in 1:(nrows * ndims_)
        data[i] = lcg_f32()
    end

    count_ref = reference(nrows, ndims_, top_k, data, label)

    d_label = CuArray(label)
    d_data = CuArray(data)
    d_count = CUDA.zeros(Int32, 1)

    CUDA.synchronize()

    ngrid = nrows ÷ 4
    while ngrid <= nrows
        @printf("Grid size is %d\n", ngrid)
        t0 = time_ns()
        for _ in 1:repeat_n
            CUDA.fill!(d_count, Int32(0))
            @cuda threads=GPU_NUM_THREADS blocks=ngrid accuracy_kernel!(
                Int32(nrows), Int32(ndims_), Int32(top_k),
                d_data, d_label, d_count)
        end
        CUDA.synchronize()
        elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
        @printf("Average execution time of accuracy kernel: %f (us)\n", elapsed_us)

        count = Array(d_count)[1]
        println(count == count_ref ? "PASS" : "FAIL")

        ngrid += nrows ÷ 4
    end
    return 0
end

main()

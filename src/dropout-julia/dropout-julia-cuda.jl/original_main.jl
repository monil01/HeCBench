using CUDA
using Printf

# Julia port of dropout-cuda.  CUDA uses Philox via cuRAND; this port uses a
# deterministic per-index hash RNG so each kernel variant remains self-checkable
# without external library state.

const BLOCK_SIZE = 256

@inline function rand01(seed::UInt64, idx::UInt64, p::UInt64, lane::UInt64)
    x = seed ⊻ (idx * UInt64(0x9e3779b97f4a7c15)) ⊻ (p * UInt64(0xbf58476d1ce4e5b9)) ⊻ lane
    x ⊻= x >> 30
    x *= UInt64(0xbf58476d1ce4e5b9)
    x ⊻= x >> 27
    x *= UInt64(0x94d049bb133111eb)
    x ⊻= x >> 31
    return Float32(x >> 40) / Float32(UInt32(1) << 24)
end

function dropout_vec1_kernel!(a, b, mask, total::Int32, p::Float32, pstep::UInt64)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x * Int32(4)
    rounded = ((total - Int32(1)) ÷ stride + Int32(1)) * stride
    scale = 1.0f0 / p
    linear = idx
    while linear < rounded
        lane = Int32(0)
        while lane < Int32(4)
            li = linear + gridDim().x * blockDim().x * lane
            if li < total
                keep = rand01(UInt64(12345678), UInt64(li), pstep, UInt64(lane)) < p
                @inbounds b[li + Int32(1)] = keep ? a[li + Int32(1)] * scale : 0.0f0
                @inbounds mask[li + Int32(1)] = keep ? UInt8(1) : UInt8(0)
            end
            lane += Int32(1)
        end
        linear += stride
    end
    return
end

function dropout_vec_kernel!(a, b, mask, total::Int32, p::Float32, pstep::UInt64, vec::Int32)
    idx = ((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)) * vec
    stride = gridDim().x * blockDim().x * vec
    scale = 1.0f0 / p
    linear = idx
    while linear < total
        lane = Int32(0)
        while lane < vec
            li = linear + lane
            if li < total
                keep = rand01(UInt64(87654321), UInt64(li), pstep, UInt64(lane)) < p
                @inbounds b[li + Int32(1)] = keep ? a[li + Int32(1)] * scale : 0.0f0
                @inbounds mask[li + Int32(1)] = keep ? UInt8(1) : UInt8(0)
            end
            lane += Int32(1)
        end
        linear += stride
    end
    return
end

function run_vec1!(d_self, d_ret, d_mask, nelem::Int, repeat_n::Int)
    CUDA.synchronize()
    t0 = time_ns()
    for p in 1:repeat_n
        pa = Float32(p / repeat_n)
        @cuda threads=BLOCK_SIZE blocks=512 dropout_vec1_kernel!(
            d_self, d_ret, d_mask, Int32(nelem), pa, UInt64(p))
    end
    CUDA.synchronize()
    @printf("Total kernel execution time (VEC1) %lf (s)\n", (time_ns() - t0) * 1e-9)
end

function run_vec!(label::Int, blocks::Int, d_self, d_ret, d_mask, nelem::Int, repeat_n::Int)
    CUDA.synchronize()
    t0 = time_ns()
    for p in 1:repeat_n
        pa = Float32(p / repeat_n)
        @cuda threads=BLOCK_SIZE blocks=blocks dropout_vec_kernel!(
            d_self, d_ret, d_mask, Int32(nelem), pa, UInt64(p), Int32(label))
    end
    CUDA.synchronize()
    @printf("Total kernel execution time (VEC%d) %lf (s)\n", label, (time_ns() - t0) * 1e-9)
end

function sanity_check(ret::Vector{Float32}, mask::Vector{UInt8}, repeat_n::Int)
    pa = Float32(1.0)
    expected = 0.1f0 / pa
    for i in eachindex(ret)
        if mask[i] > 1 || (mask[i] == 1 && ret[i] != expected) || (mask[i] == 0 && ret[i] != 0.0f0)
            return false
        end
    end
    return repeat_n > 0
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    nelem = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])

    self_info = fill(0.1f0, nelem)
    d_self = CuArray(self_info)
    d_ret = CUDA.zeros(Float32, nelem)
    d_mask = CUDA.zeros(UInt8, nelem)

    run_vec1!(d_self, d_ret, d_mask, nelem, repeat_n)
    run_vec!(2, 256, d_self, d_ret, d_mask, nelem, repeat_n)
    run_vec!(4, 128, d_self, d_ret, d_mask, nelem, repeat_n)

    ok = sanity_check(Array(d_ret), Array(d_mask), repeat_n)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())

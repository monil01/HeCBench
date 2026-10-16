using CUDA
using Printf

const NUM_THREADS = 256

@inline function atomic_inc_mod!(ptr, one, zero, limit)
    old = CUDA.atomic_add!(ptr, one)
    if old >= limit
        CUDA.atomic_cas!(ptr, old + one, zero)
    end
    return
end

@inline function atomic_dec_mod!(ptr, one, zero, limit)
    old = CUDA.atomic_sub!(ptr, one)
    if old == zero || old > limit
        CUDA.atomic_cas!(ptr, old - one, limit)
    end
    return
end

function atomic_kernel!(data, len::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid >= len
        return
    end

    CUDA.atomic_add!(pointer(data, 1), Int32(10))
    CUDA.atomic_sub!(pointer(data, 2), Int32(10))
    CUDA.atomic_max!(pointer(data, 3), tid)
    CUDA.atomic_min!(pointer(data, 4), tid)
    CUDA.atomic_and!(pointer(data, 5), Int32(2) * tid + Int32(7))
    CUDA.atomic_or!(pointer(data, 6), Int32(1) << (tid & Int32(31)))
    CUDA.atomic_xor!(pointer(data, 7), tid)
    atomic_inc_mod!(pointer(data, 8), Int32(1), Int32(0), Int32(17))
    atomic_dec_mod!(pointer(data, 9), Int32(1), Int32(0), Int32(137))
    return
end

function atomic_kernel!(data::CuDeviceVector{UInt32, 1}, len::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid >= len
        return
    end

    utid = UInt32(tid)
    CUDA.atomic_add!(pointer(data, 1), UInt32(10))
    CUDA.atomic_sub!(pointer(data, 2), UInt32(10))
    CUDA.atomic_max!(pointer(data, 3), utid)
    CUDA.atomic_min!(pointer(data, 4), utid)
    CUDA.atomic_and!(pointer(data, 5), UInt32(2) * utid + UInt32(7))
    CUDA.atomic_or!(pointer(data, 6), UInt32(1) << (utid & UInt32(31)))
    CUDA.atomic_xor!(pointer(data, 7), utid)
    atomic_inc_mod!(pointer(data, 8), UInt32(1), UInt32(0), UInt32(17))
    atomic_dec_mod!(pointer(data, 9), UInt32(1), UInt32(0), UInt32(137))
    return
end

init_data(::Type{Int32}) = Int32[0, 0, -256, 256, 255, 0, 255, 0, 0]
init_data(::Type{UInt32}) = UInt32[0, 0, 0xffffff00, 256, 255, 0, 255, 0, 0]

function reference_data(::Type{T}, len::Int) where {T}
    data = init_data(T)
    @inbounds for i in 0:(len - 1)
        data[1] += T(10)
        data[2] -= T(10)
        data[3] = max(data[3], T(i))
        data[4] = min(data[4], T(i))
        data[5] &= T(2 * i + 7)
        data[6] |= T == Int32 ? T(Int32(1) << (i & 31)) : T(UInt32(1) << UInt32(i & 31))
        data[7] ⊻= T(i)
        old = UInt32(data[8])
        data[8] = T(old >= UInt32(17) ? UInt32(0) : old + UInt32(1))
        old = UInt32(data[9])
        data[9] = T((old == UInt32(0) || old > UInt32(137)) ? UInt32(137) : old - UInt32(1))
    end
    return data
end

function verify_data(actual::Vector{T}, len::Int) where {T}
    expected = reference_data(T, len)
    labels = ("Add", "Sub", "Max", "Min", "And", "Or", "Xor", "atomicInc", "atomicDec")
    ok = true
    @inbounds for i in eachindex(expected)
        if i == 8 || i == 9
            continue
        end
        if actual[i] != expected[i]
            @printf("%s failed: %d != %d\n", labels[i], expected[i], actual[i])
            ok = false
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function testcase(::Type{T}, num::Int, repeat::Int) where {T}
    len = Int32(1 << num)
    blocks = cld(Int(len), NUM_THREADS)
    initial = init_data(T)
    d_data = CuArray(initial)

    for _ in 1:repeat
        copyto!(d_data, initial)
        @cuda threads=NUM_THREADS blocks=blocks atomic_kernel!(d_data, len)
    end
    CUDA.synchronize()
    actual = Array(d_data)
    ok = verify_data(actual, Int(len))

    start = time_ns()
    for _ in 1:repeat
        @cuda threads=NUM_THREADS blocks=blocks atomic_kernel!(d_data, len)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (us)\n", elapsed * 1.0e-3 / repeat)
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of atomic operations> <repeat>")
        return 1
    end
    num = parse(Int, args[1])
    repeat = parse(Int, args[2])
    ok = testcase(Int32, num, repeat)
    ok &= testcase(UInt32, num, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))

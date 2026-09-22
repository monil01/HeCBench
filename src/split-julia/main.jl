using CUDA
using Printf

const WARP_SIZE = Int32(32)
const THREADS = 128
const KEYS_PER_BLOCK = 4 * THREADS

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function split_sort_kernel!(keys_in, keys_out, nblocks::Int32)
    bid = blockIdx().x
    tid = threadIdx().x
    base = (bid - Int32(1)) * Int32(KEYS_PER_BLOCK)

    sh = @cuDynamicSharedMem(UInt32, KEYS_PER_BLOCK + 16)
    i0 = (tid - Int32(1)) * Int32(4)
    @inbounds begin
        sh[i0 + Int32(1)] = keys_in[base + i0 + Int32(1)]
        sh[i0 + Int32(2)] = keys_in[base + i0 + Int32(2)]
        sh[i0 + Int32(3)] = keys_in[base + i0 + Int32(3)]
        sh[i0 + Int32(4)] = keys_in[base + i0 + Int32(4)]
    end
    sync_threads()

    if tid == Int32(1)
        i = Int32(1)
        while i <= Int32(16)
            @inbounds sh[Int32(KEYS_PER_BLOCK) + i] = UInt32(0)
            i += Int32(1)
        end

        i = Int32(1)
        while i <= Int32(KEYS_PER_BLOCK)
            v = @inbounds sh[i]
            count_idx = Int32(KEYS_PER_BLOCK) + Int32(v) + Int32(1)
            @inbounds sh[count_idx] += UInt32(1)
            i += Int32(1)
        end

        pos = Int32(1)
        bucket = UInt32(0)
        while bucket <= UInt32(15)
            c = @inbounds sh[Int32(KEYS_PER_BLOCK) + Int32(bucket) + Int32(1)]
            j = UInt32(0)
            while j < c
                @inbounds sh[pos] = bucket
                pos += Int32(1)
                j += UInt32(1)
            end
            bucket += UInt32(1)
        end
    end
    sync_threads()

    @inbounds begin
        keys_out[base + i0 + Int32(1)] = sh[i0 + Int32(1)]
        keys_out[base + i0 + Int32(2)] = sh[i0 + Int32(2)]
        keys_out[base + i0 + Int32(3)] = sh[i0 + Int32(3)]
        keys_out[base + i0 + Int32(4)] = sh[i0 + Int32(4)]
    end
    return
end

function verify(sorted_keys::Vector{UInt32}, keys::Vector{UInt32}, threads::Int, n::Int)
    m1 = zeros(Int, 16)
    m2 = zeros(Int, 16)
    block_n = threads * 4

    @inbounds for i in 1:block_n:n
        for j in 0:block_n-2
            if sorted_keys[i + j] > sorted_keys[i + j + 1]
                return false
            end
        end
    end

    @inbounds for v in sorted_keys
        if v >= UInt32(16)
            return false
        end
    end

    @inbounds for i in 1:block_n:n
        fill!(m1, 0)
        fill!(m2, 0)
        for j in 0:block_n-1
            m1[Int(keys[i + j]) + 1] += 1
            m2[Int(sorted_keys[i + j]) + 1] += 1
        end
        if m1 != m2
            return false
        end
    end
    return true
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of keys> <repeat>")
        return 1
    end
    n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    if n % KEYS_PER_BLOCK != 0
        println("number of keys must be a multiple of $KEYS_PER_BLOCK")
        return 1
    end

    libc_srand(512)
    keys = Vector{UInt32}(undef, n)
    @inbounds for i in 1:n
        keys[i] = UInt32(mod(libc_rand(), 16))
    end

    d_keys = CuArray(keys)
    d_out = CUDA.zeros(UInt32, n)
    teams = div(n, KEYS_PER_BLOCK)
    shmem = (KEYS_PER_BLOCK + 16) * sizeof(UInt32)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=teams shmem=shmem split_sort_kernel!(d_keys, d_out, Int32(teams))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (us)\n", elapsed * 1.0e-3 / repeat)

    out = Array(d_out)
    println(verify(out, keys, THREADS, n) ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))

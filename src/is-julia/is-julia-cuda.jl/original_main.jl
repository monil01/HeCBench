using CUDA
using Printf

const CLASS = 'S'
const TOTAL_KEYS_LOG_2 = 16
const MAX_KEY_LOG_2 = 11
const TOTAL_KEYS = 1 << TOTAL_KEYS_LOG_2
const MAX_KEY = 1 << MAX_KEY_LOG_2
const NUM_KEYS = TOTAL_KEYS
const SIZE_OF_BUFFERS = NUM_KEYS
const MAX_ITERATIONS = 24
const TEST_ARRAY_SIZE = 5

const TEST_INDEX = Int32[48427, 17148, 23627, 62548, 4431]
const TEST_RANK = Int32[0, 18, 346, 64917, 65463]

const R23 = 0.5^23
const R46 = R23 * R23
const T23 = 2.0^23
const T46 = T23 * T23

function randlc!(x::Base.RefValue{Float64}, a::Base.RefValue{Float64})
    t1 = R23 * a[]
    j = floor(Int, t1)
    a1 = Float64(j)
    a2 = a[] - T23 * a1
    t1 = R23 * x[]
    j = floor(Int, t1)
    x1 = Float64(j)
    x2 = x[] - T23 * x1
    t1 = a1 * x2 + a2 * x1
    j = floor(Int, R23 * t1)
    t2 = Float64(j)
    z = t1 - T23 * t2
    t3 = T23 * z + a2 * x2
    j = floor(Int, R46 * t3)
    t4 = Float64(j)
    x[] = t3 - T46 * t4
    return R46 * x[]
end

function find_my_seed(kn::Int, np::Int, nn::Int, s::Float64, a::Float64)
    kn == 0 && return s
    mq = cld(nn ÷ 4, np)
    nq = mq * 4 * kn
    t1 = Ref(s)
    t2 = Ref(a)
    kk = nq
    while kk > 1
        ik = kk ÷ 2
        if 2 * ik == kk
            randlc!(t2, t2)
            kk = ik
        else
            randlc!(t1, t2)
            kk -= 1
        end
    end
    randlc!(t1, t2)
    return t1[]
end

function create_seq(threads_per_block::Int)
    amount_of_work = threads_per_block * threads_per_block
    key_array = Vector{Int32}(undef, SIZE_OF_BUFFERS)
    mq = cld(NUM_KEYS, amount_of_work)
    scale = MAX_KEY ÷ 4
    for myid in 0:amount_of_work-1
        k1 = mq * myid
        k2 = min(k1 + mq, NUM_KEYS)
        s = Ref(find_my_seed(myid, amount_of_work, 4 * NUM_KEYS, 314159265.0, 1220703125.0))
        an = Ref(1220703125.0)
        for i in k1:k2-1
            x = randlc!(s, an)
            x += randlc!(s, an)
            x += randlc!(s, an)
            x += randlc!(s, an)
            key_array[i + 1] = Int32(floor(scale * x))
        end
    end
    return key_array
end

function histogram_kernel!(hist, keys, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds key = keys[i] + Int32(1)
        CUDA.@atomic hist[key] += Int32(1)
    end
    return
end

function set_iteration_keys_kernel!(keys, iteration::Int32)
    if threadIdx().x == Int32(1)
        keys[iteration + Int32(1)] = iteration
        keys[iteration + Int32(MAX_ITERATIONS) + Int32(1)] = Int32(MAX_KEY) - iteration
    end
    return
end

function partial_verify(hist::Vector{Int32}, vals::Vector{Int32}, iteration::Int)
    passed = 0
    for i in 1:TEST_ARRAY_SIZE
        k = vals[i]
        if 0 < k <= NUM_KEYS - 1
            key_rank = hist[Int(k)]
            expected = i <= 3 ? TEST_RANK[i] + iteration : TEST_RANK[i] - iteration
            if key_rank == expected
                passed += 1
            end
        end
    end
    return passed
end

function main()
    println("\n\n NAS Parallel Benchmarks 4.1 IS Benchmark\n")
    println(" Size:  ", TOTAL_KEYS, "  (class ", CLASS, ")")
    println(" Iterations:   ", MAX_ITERATIONS)

    if length(ARGS) != 3
        println("Usage: $(PROGRAM_FILE) <threads per block for the create_seq kernel>")
        println("           <threads per block for the rank kernel>")
        println("           <threads per block for the verify kernel>")
        exit(1)
    end

    threads_create = parse(Int, ARGS[1])
    threads_rank = parse(Int, ARGS[2])

    key_array = create_seq(threads_create)
    d_keys = CuArray(key_array)
    d_hist = CUDA.zeros(Int32, MAX_KEY)
    blocks_keys = cld(NUM_KEYS, threads_rank)

    passed_verification = 0
    CUDA.synchronize()
    start = time_ns()

    for iteration in 1:MAX_ITERATIONS
        @cuda threads=1 blocks=1 set_iteration_keys_kernel!(d_keys, Int32(iteration))
        CUDA.fill!(d_hist, Int32(0))
        @cuda threads=threads_rank blocks=blocks_keys histogram_kernel!(d_hist, d_keys, Int32(NUM_KEYS))
        d_prefix = accumulate(+, d_hist)
        hist = Array(d_prefix)
        vals = Array(d_keys)[Int.(TEST_INDEX) .+ 1]
        passed_verification += partial_verify(hist, vals, iteration)
        d_hist = d_prefix
    end

    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9 / MAX_ITERATIONS
    @printf("Average execution time of the rank kernels %f (s)\n", elapsed_s)

    final_keys = Array(d_keys)
    sorted_keys = sort(final_keys)
    out_of_sort = count(i -> sorted_keys[i - 1] > sorted_keys[i], 2:length(sorted_keys))
    if out_of_sort != 0
        println("Full_verify: number of keys out of sort: ", out_of_sort)
    else
        passed_verification += 1
    end

    if passed_verification != 5 * MAX_ITERATIONS + 1
        passed_verification = 0
    end
    println(passed_verification != 0 ? "PASS" : "FAIL")
end

main()

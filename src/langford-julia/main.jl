using Printf

const K_MAX_N = 31
const K_NUM_LOGICAL_THREADS = 16383
const LSB = Int64(1)
const LSB32 = Int32(1)

ffsl(x::Int64) = x == 0 ? 0 : trailing_zeros(reinterpret(UInt64, x)) + 1
unixtime_ms() = round(Int64, time() * 1000)

function init_known_results()
    known = zeros(Int64, 64)
    for i in 29:63
        if i % 4 == 3 || i % 4 == 0
            known[i + 1] = -1
        end
    end
    known[3 + 1] = 1
    known[4 + 1] = 0
    known[7 + 1] = 0
    known[8 + 1] = 4
    known[11 + 1] = 16
    known[12 + 1] = 40
    known[15 + 1] = 194
    known[16 + 1] = 274
    known[19 + 1] = 2384
    known[20 + 1] = 4719
    known[23 + 1] = 31856
    known[24 + 1] = 62124
    known[27 + 1] = 426502
    known[28 + 1] = 817717
    return known
end

function push_state!(stack, top::Int, k::Int8, m::Int8, d::Int8, num_open::Int8)
    stack[top + 1] = k
    stack[top + 2] = m
    stack[top + 3] = d
    stack[top + 4] = num_open
    return top + 4
end

function enumerate_logical!(results::Vector{Vector{Int8}}, n::Int, logical_thread_index::Int)
    two_n = 2 * n
    msb = LSB << (n - 1)
    nn1 = LSB << (2 * n - 1)
    pos = zeros(Int8, n)
    availability = zeros(Int32, 2 * n + 1)
    open = zeros(Int64, 4 * n + 2)
    stack = zeros(Int8, 24 * n)

    availability[1] = Int32(msb | (msb - 1))
    open[1] = 0
    open[2] = 0
    top = push_state!(stack, 0, Int8(0), Int8(-1), Int8(0), Int8(0))

    while top > 0
        num_open = stack[top]; top -= 1
        d = stack[top]; top -= 1
        m = stack[top]; top -= 1
        k = stack[top]; top -= 1

        kk = Int(k)
        open[2 * kk + 3] = open[2 * kk + 1]
        open[2 * kk + 4] = open[2 * kk + 2]
        avail = availability[kk + 1]

        if m >= 0
            pos[Int(m) + 1] = k
            avail = xor(avail, Int32(LSB32 << Int(m)))
            idx = 2 * kk + 3 + Int(d)
            open[idx] &= open[idx] - 1
        else
            idx = 2 * kk + 3 + Int(d)
            open[idx] |= nn1 >> kk
            num_open += Int8(1)
        end

        kk += 1
        availability[kk + 1] = avail

        if kk == two_n
            push!(results, copy(pos))
            continue
        end

        k_limit = n > 19 ? 8 + (n ÷ 3) : n - 5
        if K_NUM_LOGICAL_THREADS > 1 && kk == k_limit
            diff_bits = reinterpret(UInt64, Int64(open[2 * (kk - 1) + 4] - open[2 * (kk - 1) + 3]))
            h = UInt64(131071) * diff_bits + UInt64(reinterpret(UInt32, avail))
            if Int(h % UInt64(K_NUM_LOGICAL_THREADS)) != logical_thread_index
                continue
            end
        end

        offset = kk - two_n - 2
        for dd in 0:1
            openings = open[2 * (kk - 1) + 3 + dd]
            if openings != 0
                mm = offset + ffsl(openings)
                if 0 <= mm < n && ((avail >> mm) & 1) == 1
                    if mm != 0 || kk <= n
                        top = push_state!(stack, top, Int8(kk), Int8(mm), Int8(dd), num_open)
                    end
                end
            end
        end

        if num_open < n
            top = push_state!(stack, top, Int8(kk), Int8(-1), Int8(1), num_open)
            top = push_state!(stack, top, Int8(kk), Int8(-1), Int8(0), num_open)
        end
    end
end

function run_case(n::Int, known_results)
    println()
    println("------")
    println("$(unixtime_ms()) Computing PL(2, $n)")
    if n > K_MAX_N
        println("$(unixtime_ms()) Sorry, n = $n exceeds the max allowed $K_MAX_N")
        return
    end

    results = Vector{Vector{Int8}}()
    t0 = time_ns()
    for logical_thread_index in 0:K_NUM_LOGICAL_THREADS-1
        enumerate_logical!(results, n, logical_thread_index)
    end
    elapsed_s = (time_ns() - t0) * 1e-9
    @printf("Kernel execution time:  %.6f (s)\n", elapsed_s)

    sort!(results)
    total = isempty(results) ? 0 : 1
    for i in 2:length(results)
        if results[i] != results[i - 1]
            total += 1
        end
    end

    print("$(unixtime_ms()) Result $total for n = $n")
    known = (0 <= n < 64) ? known_results[n + 1] : Int64(-1)
    if n < 0 || n >= 64 || known == -1
        print(" is NEW")
    elseif known == total
        print(" MATCHES previously published result")
    else
        print(" MISMATCHES previously published result $known")
    end
    println()
    println("------")
    println()
end

function main()
    known_results = init_known_results()
    for n in (7, 8, 11, 12, 15)
        run_case(n, known_results)
    end
end

main()

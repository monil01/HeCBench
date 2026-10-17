using CUDA
using Printf

const A = 0.0
const B = 15.0
const ROW_SIZE = 17
const EPS = 1.0e-7

@inline f(x) = exp(x) * sin(x)

function romberg_kernel!(result, a::Float64, b::Float64, nwg::Int32)
    bid = blockIdx().x
    if threadIdx().x != Int32(1)
        return
    end

    diff = (b - a) / Float64(nwg)
    lo = a + Float64(bid - Int32(1)) * diff
    hi = lo + diff
    max_eval = Int32(1 << (ROW_SIZE - 1))
    step = (hi - lo) / Float64(max_eval)

    table = ntuple(_ -> 0.0, Val(ROW_SIZE))
    local_col = Base.setindex(table, f(lo) + f(hi), 1)

    k = Int32(1)
    while k < max_eval
        pos = ROW_SIZE - trailing_zeros(k)
        local_col = Base.setindex(local_col, local_col[pos] + 2.0 * f(lo + step * Float64(k)), pos)
        k += Int32(1)
    end

    vals = local_col
    for i in 2:ROW_SIZE
        vals = Base.setindex(vals, vals[i - 1] + vals[i], i)
    end
    for i in 1:ROW_SIZE
        vals = Base.setindex(vals, vals[i] * (hi - lo) / Float64(1 << i), i)
    end

    for col in 1:(ROW_SIZE - 1)
        row = ROW_SIZE
        while row > col
            v = vals[row] + (vals[row] - vals[row - 1]) / Float64((1 << (2 * col - 1)) - 1)
            vals = Base.setindex(vals, v, row)
            row -= 1
        end
    end

    @inbounds result[bid] = vals[ROW_SIZE]
    return
end

function reference_romberg(a::Float64, b::Float64, max_steps::Int, acc::Float64)
    rp = zeros(Float64, max_steps)
    rc = zeros(Float64, max_steps)
    h = b - a
    rp[1] = (f(a) + f(b)) * h * 0.5
    for i in 2:max_steps
        h /= 2.0
        c = 0.0
        ep = 1 << (i - 2)
        for j in 1:ep
            c += f(a + (2 * j - 1) * h)
        end
        rc[1] = h * c + 0.5 * rp[1]
        for j in 2:i
            nk = 4.0^(j - 1)
            rc[j] = (nk * rc[j - 1] - rp[j - 1]) / (nk - 1.0)
        end
        if i > 2 && abs(rp[i - 1] - rc[i]) < acc
            return rc[i - 1]
        end
        rp, rc = rc, rp
    end
    return rp[max_steps]
end

function main(args)
    if length(args) != 3
        println("Usage: ./main <number of work-groups> <work-group size> <repeat>")
        return 1
    end
    nwg = parse(Int, args[1])
    wgs = parse(Int, args[2])
    repeat = parse(Int, args[3])
    d_result = CUDA.zeros(Float64, nwg)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=wgs blocks=nwg romberg_kernel!(d_result, A, B, Int32(nwg))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9 / repeat
    @printf("Average kernel execution time: %f (s)\n", elapsed_s)

    total = sum(Array(d_result))
    ref_sum = reference_romberg(A, B, ROW_SIZE, EPS)
    ok = abs(total - ref_sum) <= EPS
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))

using CUDA
using Printf

const FILES = [
    "../extend2-sycl/testdata/extend2-qlen102-tlen297-tsc387439.bin",
    "../extend2-sycl/testdata/extend2-qlen30-tlen55-tsc36010.bin",
    "../extend2-sycl/testdata/extend2-qlen10-tlen210-tsc7696.bin",
    "../extend2-sycl/testdata/extend2-qlen44-tlen227-tsc76141.bin",
    "../extend2-sycl/testdata/extend2-qlen110-tlen310-tsc401739.bin",
    "../extend2-sycl/testdata/extend2-qlen55-tlen181-tsc80496.bin",
    "../extend2-sycl/testdata/extend2-qlen112-tlen291-tsc10951304.bin",
    "../extend2-sycl/testdata/extend2-qlen66-tlen226-tsc149708.bin",
    "../extend2-sycl/testdata/extend2-qlen146-tlen346-tsc547625.bin",
    "../extend2-sycl/testdata/extend2-qlen74-tlen251-tsc200148.bin",
    "../extend2-sycl/testdata/extend2-qlen148-tlen291-tsc426764.bin",
    "../extend2-sycl/testdata/extend2-qlen7-tlen207-tsc4823.bin",
    "../extend2-sycl/testdata/extend2-qlen175-tlen375-tsc654888.bin",
    "../extend2-sycl/testdata/extend2-qlen80-tlen280-tsc236327.bin",
    "../extend2-sycl/testdata/extend2-qlen189-tlen373-tsc650936.bin",
    "../extend2-sycl/testdata/extend2-qlen81-tlen281-tsc256607.bin",
    "../extend2-sycl/testdata/extend2-qlen26-tlen212-tsc28860.bin",
]

struct Extend2Dat
    qlen::Int32
    query::Vector{UInt8}
    tlen::Int32
    target::Vector{UInt8}
    m::Int32
    mat::Vector{Int8}
    o_del::Int32
    e_del::Int32
    o_ins::Int32
    e_ins::Int32
    w::Int32
    end_bonus::Int32
    zdrop::Int32
    h0::Int32
    qle::Int32
    tle::Int32
    gtle::Int32
    gscore::Int32
    max_off::Int32
    score::Int32
end

read_i32(io) = read(io, Int32)

function read_data(path::String)
    open(path, "r") do io
        qlen = read_i32(io)
        query = read(io, qlen)
        tlen = read_i32(io)
        target = read(io, tlen)
        m = read_i32(io)
        mat = reinterpret(Int8, read(io, 25))
        o_del = read_i32(io)
        e_del = read_i32(io)
        o_ins = read_i32(io)
        e_ins = read_i32(io)
        w = read_i32(io)
        end_bonus = read_i32(io)
        zdrop = read_i32(io)
        h0 = read_i32(io)
        qle = read_i32(io)
        tle = read_i32(io)
        gtle = read_i32(io)
        gscore = read_i32(io)
        max_off = read_i32(io)
        score = read_i32(io)
        read(io, UInt64)
        read(io, Float64)
        return Extend2Dat(qlen, query, tlen, target, m, collect(mat),
                          o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
                          qle, tle, gtle, gscore, max_off, score)
    end
end

function kernel_extend2!(query, target, mat, eh_h, eh_e, qp, out,
                         qlen::Int32, tlen::Int32, m_arg::Int32,
                         o_del::Int32, e_del::Int32, o_ins::Int32, e_ins::Int32,
                         w_in::Int32, end_bonus::Int32, zdrop::Int32, h0::Int32)
    oe_del = o_del + e_del
    oe_ins = o_ins + e_ins

    idx = Int32(1)
    for k = Int32(0):(m_arg - Int32(1))
        base = k * m_arg
        for j = Int32(0):(qlen - Int32(1))
            qp[idx] = mat[base + Int32(query[j + Int32(1)]) + Int32(1)]
            idx += Int32(1)
        end
    end

    eh_h[Int32(1)] = h0
    eh_h[Int32(2)] = h0 > oe_ins ? h0 - oe_ins : Int32(0)
    j = Int32(2)
    while j <= qlen && eh_h[j] > e_ins
        eh_h[j + Int32(1)] = eh_h[j] - e_ins
        j += Int32(1)
    end

    kmax = m_arg * m_arg
    max_score_mat = Int32(0)
    for p = Int32(1):kmax
        v = Int32(mat[p])
        max_score_mat = max_score_mat > v ? max_score_mat : v
    end
    max_ins = Int32(trunc(Float32(qlen * max_score_mat + end_bonus - o_ins) / Float32(e_ins) + 1.0f0))
    max_ins = max_ins > Int32(1) ? max_ins : Int32(1)
    w = w_in < max_ins ? w_in : max_ins
    max_del = Int32(trunc(Float32(qlen * max_score_mat + end_bonus - o_del) / Float32(e_del) + 1.0f0))
    max_del = max_del > Int32(1) ? max_del : Int32(1)
    w = w < max_del ? w : max_del

    maxv = h0
    max_i = Int32(-1)
    max_j = Int32(-1)
    max_ie = Int32(-1)
    gscore = Int32(-1)
    max_off = Int32(0)
    beg = Int32(0)
    endv = qlen
    i = Int32(0)
    while i < tlen
        f = Int32(0)
        mrow = Int32(0)
        mj = Int32(-1)
        qbase = Int32(target[i + Int32(1)]) * qlen

        if beg < i - w
            beg = i - w
        end
        if endv > i + w + Int32(1)
            endv = i + w + Int32(1)
        end
        if endv > qlen
            endv = qlen
        end

        h1 = beg == Int32(0) ? h0 - (o_del + e_del * (i + Int32(1))) : Int32(0)
        if h1 < Int32(0)
            h1 = Int32(0)
        end

        j = beg
        while j < endv
            arr = j + Int32(1)
            M = eh_h[arr]
            e = eh_e[arr]
            eh_h[arr] = h1
            M = M != Int32(0) ? M + Int32(qp[qbase + j + Int32(1)]) : Int32(0)
            h = M > e ? M : e
            h = h > f ? h : f
            h1 = h
            mj = mrow > h ? mj : j
            mrow = mrow > h ? mrow : h
            t = M - oe_del
            t = t > Int32(0) ? t : Int32(0)
            e -= e_del
            e = e > t ? e : t
            eh_e[arr] = e
            t = M - oe_ins
            t = t > Int32(0) ? t : Int32(0)
            f -= e_ins
            f = f > t ? f : t
            j += Int32(1)
        end
        eh_h[endv + Int32(1)] = h1
        eh_e[endv + Int32(1)] = Int32(0)
        if j == qlen
            max_ie = gscore > h1 ? max_ie : i
            gscore = gscore > h1 ? gscore : h1
        end
        if mrow == Int32(0)
            break
        end
        if mrow > maxv
            maxv = mrow
            max_i = i
            max_j = mj
            delta = mj > i ? mj - i : i - mj
            max_off = max_off > delta ? max_off : delta
        elseif zdrop > Int32(0)
            if i - max_i > mj - max_j
                if maxv - mrow - ((i - max_i) - (mj - max_j)) * e_del > zdrop
                    break
                end
            else
                if maxv - mrow - ((mj - max_j) - (i - max_i)) * e_ins > zdrop
                    break
                end
            end
        end

        j = beg
        while j < endv && eh_h[j + Int32(1)] == Int32(0) && eh_e[j + Int32(1)] == Int32(0)
            j += Int32(1)
        end
        beg = j
        j = endv
        while j >= beg && eh_h[j + Int32(1)] == Int32(0) && eh_e[j + Int32(1)] == Int32(0)
            j -= Int32(1)
        end
        endv = j + Int32(2) < qlen ? j + Int32(2) : qlen
        i += Int32(1)
    end

    out[Int32(1)] = max_j + Int32(1)
    out[Int32(2)] = max_i + Int32(1)
    out[Int32(3)] = max_ie + Int32(1)
    out[Int32(4)] = gscore
    out[Int32(5)] = max_off
    out[Int32(6)] = maxv
    return
end

function extend2(d::Extend2Dat)
    d_query = CuArray(d.query)
    d_target = CuArray(d.target)
    d_mat = CuArray(d.mat)
    d_eh_h = CUDA.zeros(Int32, Int(d.qlen) + 1)
    d_eh_e = CUDA.zeros(Int32, Int(d.qlen) + 1)
    d_qp = CUDA.zeros(Int8, Int(d.qlen * d.m))
    d_out = CUDA.zeros(Int32, 6)

    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=1 blocks=1 kernel_extend2!(d_query, d_target, d_mat, d_eh_h, d_eh_e, d_qp, d_out,
                                             d.qlen, d.tlen, d.m, d.o_del, d.e_del, d.o_ins,
                                             d.e_ins, d.w, d.end_bonus, d.zdrop, d.h0)
    CUDA.synchronize()
    elapsed_ns = time_ns() - t0

    out = Array(d_out)
    ok = out == Int32[d.qle, d.tle, d.gtle, d.gscore, d.max_off, d.score]
    if !ok
        labels = ("qle", "tle", "gtle", "gscore", "max_off", "score")
        expected = Int32[d.qle, d.tle, d.gtle, d.gscore, d.max_off, d.score]
        for n in eachindex(out)
            if out[n] != expected[n]
                @printf("Error: %s %d %d\n", labels[n], expected[n], out[n])
            end
        end
    end
    return elapsed_ns, ok
end

function main()
    length(ARGS) == 1 || error("Usage: julia main.jl <repeat>")
    repeat = parse(Int, ARGS[1])
    total_ns = 0
    all_ok = true
    for f = 1:repeat
        d = read_data(FILES[mod1(f, length(FILES))])
        elapsed_ns, ok = extend2(d)
        total_ns += elapsed_ns
        all_ok &= ok
    end
    @printf("Average offload time %f (us)\n", total_ns * 1e-3 / repeat)
    println(all_ok ? "PASS" : "FAIL")
end

main()

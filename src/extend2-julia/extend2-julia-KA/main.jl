using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
@kernel function kernel_extend2!(query, target, mat, eh_h, eh_e, qp, out,
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
end

function launch(q,t,m,hh,ee,qp,out,params)
 kernel_extend2!(BACKEND,1)(q,t,m,hh,ee,qp,out,params...;ndrange=1)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)

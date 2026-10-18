using StaticArrays
const RD=8
const RD2=64
@inline function matvec!(A, v, out)
    @inbounds for i in 1:RD
        s = 0.0
        for j in 1:RD
            s += A[i + (j - 1) * RD] * v[j]
        end
        out[i] = s
    end
    return
end

@inline function matvec!(alpha, A, v, out)
    @inbounds for i in 1:RD
        s = 0.0
        for j in 1:RD
            s += A[i + (j - 1) * RD] * v[j]
        end
        out[i] = alpha * s
    end
    return
end

@inline function matmul!(A, B, out; aT=false, bT=false)
    @inbounds for i in 1:RD
        for j in 1:RD
            s = 0.0
            for k in 1:RD
                aik = aT ? A[k + (i - 1) * RD] : A[i + (k - 1) * RD]
                bkj = bT ? B[j + (k - 1) * RD] : B[k + (j - 1) * RD]
                s += aik * bkj
            end
            out[i + (j - 1) * RD] = s
        end
    end
    return
end

@inline function kalman_site!(bid0,ys,nobs::Int32,T,Z,RQR,P,alpha,intercept,mu,batch_size::Int32,vs,Fs,sumlogs,n_diff::Int32,fc_steps::Int32,fc,conf_int,F_fc)
    if bid0 >= batch_size
        return
    end
    l_RQR = MVector{RD2, Float64}(undef)
    l_T = MVector{RD2, Float64}(undef)
    l_Z = MVector{RD, Float64}(ntuple(_ -> 0.0, Val(RD)))
    l_P = MVector{RD2, Float64}(undef)
    l_alpha = MVector{RD, Float64}(undef)
    l_K = MVector{RD, Float64}(undef)
    l_tmp = MVector{RD2, Float64}(undef)
    l_TP = MVector{RD2, Float64}(undef)
    b_rd = bid0 * Int32(RD)
    b_rd2 = bid0 * Int32(RD2)
    @inbounds for i in 1:RD2
        l_RQR[i] = RQR[b_rd2 + Int32(i)]
        l_T[i] = T[b_rd2 + Int32(i)]
        l_P[i] = P[b_rd2 + Int32(i)]
    end
    @inbounds for i in 1:RD
        if n_diff > 0
            l_Z[i] = Z[b_rd + Int32(i)]
        end
        l_alpha[i] = alpha[b_rd + Int32(i)]
    end
    base_obs = bid0 * nobs
    m = intercept ? mu[bid0 + Int32(1)] : 0.0
    bsum=0.0
    @inbounds for it in Int32(0):(nobs - Int32(1))
        vs_it = ys[base_obs + it + Int32(1)]
        if n_diff == 0
            vs_it -= l_alpha[1]
        else
            for i in 1:RD
                vs_it -= l_alpha[i] * l_Z[i]
            end
        end
        vs[base_obs+it+Int32(1)]=vs_it
        f = 0.0
        if n_diff == 0
            f = l_P[1]
        else
            for i in 1:RD
                for j in 1:RD
                    f += l_P[i + (j - 1) * RD] * l_Z[i] * l_Z[j]
                end
            end
        end
        Fs[base_obs+it+Int32(1)]=f
        if it>=n_diff;bsum+=log(f);end
        matmul!(l_T, l_P, l_TP)
        invf = 1.0 / f
        if n_diff == 0
            for i in 1:RD
                l_K[i] = invf * l_TP[i]
            end
        else
            matvec!(invf, l_TP, l_Z, l_K)
        end
        matvec!(l_T, l_alpha, l_tmp)
        for i in 1:RD
            l_alpha[i] = l_tmp[i] + l_K[i] * vs_it
        end
        l_alpha[Int(n_diff) + 1] += m
        for i in 1:RD2
            l_tmp[i] = l_T[i]
        end
        if n_diff == 0
            for i in 1:RD
                l_tmp[i] -= l_K[i]
            end
        else
            for i in 1:RD
                for j in 1:RD
                    l_tmp[i + (j - 1) * RD] -= l_K[i] * l_Z[j]
                end
            end
        end
        matmul!(l_TP, l_tmp, l_P; bT=true)
        for i in 1:RD2
            l_P[i] += l_RQR[i]
        end
    end
    sumlogs[bid0+Int32(1)]=bsum
    fc_base = bid0 * fc_steps
    @inbounds for it in Int32(0):(fc_steps-Int32(1))
        pred=0.0
        if n_diff==0
            pred=l_alpha[1]
        else
            for i in 1:RD;pred+=l_alpha[i]*l_Z[i];end
        end
        fc[fc_base+it+Int32(1)]=pred
        matvec!(l_T,l_alpha,l_tmp)
        for i in 1:RD;l_alpha[i]=l_tmp[i];end
        l_alpha[Int(n_diff)+1]+=m
        if conf_int
            f=0.0
            if n_diff==0
                f=l_P[1]
            else
                for i in 1:RD,j in 1:RD;f+=l_P[i+(j-1)*RD]*l_Z[i]*l_Z[j];end
            end
            F_fc[fc_base+it+Int32(1)]=f
            matmul!(l_T,l_P,l_TP)
            matmul!(l_TP,l_T,l_P;bT=true)
            for i in 1:RD2;l_P[i]+=l_RQR[i];end
        end
    end
    return
end

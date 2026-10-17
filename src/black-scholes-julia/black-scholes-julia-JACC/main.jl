using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function bs_jacc!(index,ty,spot,strike,div,risk,T,vol,out,N)
    i=index
    @inbounds if i <= N
        S = spot[i]; K = strike[i]; q = div[i]; r = risk[i]
        Tt = T[i]; v = vol[i]
        sqrtT = sqrt(Tt)
        vs = v * sqrtT
        d1 = (log(S / K) + (r - q + 0.5f0 * v * v) * Tt) / vs
        d2 = d1 - vs
        inv_sqrt2 = 0.70710678f0
        Nd1 = 0.5f0 * (1.0f0 + erf_approx(d1 * inv_sqrt2))
        Nd2 = 0.5f0 * (1.0f0 + erf_approx(d2 * inv_sqrt2))
        eqT = exp(-q * Tt)
        erT = exp(-r * Tt)
        call = S * eqT * Nd1 - K * erT * Nd2
        put  = call - S * eqT + K * erT
        out[i] = ty[i] == Int32(1) ? put : call
    end
    return
end
bs_study((ty,S,K,q,r,T,v,out,n)->()->JACC.parallel_for(JACC.launch_spec(;threads=256,blocks=cld(Int(n),256),shmem_size=0,sync=false),Int(n),bs_jacc!,ty,S,K,q,r,T,v,out,n))

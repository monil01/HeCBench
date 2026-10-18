using CUDA,KernelAbstractions
function softplus_f(x::Float32)
    return x > 20f0 ? x : log(1f0 + exp(x))
end

function silu_f(x::Float32)
    return x / (1f0 + exp(-x))
end

@kernel unsafe_indices=true function portable_scan!(u, delta, a, bmat, cmat, dvec, delta_bias,
                                     z, ssm_states, y, batch::Int32,
                                     dim::Int32, dstate::Int32,
                                     seqlen::Int32)
    gi = @index(Global, NTuple)
    d = Int32(gi[1]);bidx=Int32(gi[2])
    if bidx <= batch && d <= dim

    state_base = (bidx - Int32(1)) * dim * dstate + (d - Int32(1)) * dstate
    h1 = ssm_states[state_base + Int32(1)]
    h2 = ssm_states[state_base + Int32(2)]
    h3 = ssm_states[state_base + Int32(3)]
    h4 = ssm_states[state_base + Int32(4)]
    h5 = ssm_states[state_base + Int32(5)]
    h6 = ssm_states[state_base + Int32(6)]
    h7 = ssm_states[state_base + Int32(7)]
    h8 = ssm_states[state_base + Int32(8)]

    row_base = (bidx - Int32(1)) * dim * seqlen + (d - Int32(1)) * seqlen
    bc_base = (bidx - Int32(1)) * dstate * seqlen
    a_base = (d - Int32(1)) * dstate
    db = delta_bias[d]
    dv = dvec[d]

    l = Int32(1)
    while l <= seqlen
        uv = u[row_base + l]
        dt = softplus_f(delta[row_base + l] + db)
        yv = dv * uv

        da = exp(dt * a[a_base + Int32(1)])
        dbv = dt * bmat[bc_base + l]
        h1 = da * h1 + dbv * uv
        yv += cmat[bc_base + l] * h1

        da = exp(dt * a[a_base + Int32(2)])
        dbv = dt * bmat[bc_base + seqlen + l]
        h2 = da * h2 + dbv * uv
        yv += cmat[bc_base + seqlen + l] * h2

        da = exp(dt * a[a_base + Int32(3)])
        dbv = dt * bmat[bc_base + Int32(2) * seqlen + l]
        h3 = da * h3 + dbv * uv
        yv += cmat[bc_base + Int32(2) * seqlen + l] * h3

        da = exp(dt * a[a_base + Int32(4)])
        dbv = dt * bmat[bc_base + Int32(3) * seqlen + l]
        h4 = da * h4 + dbv * uv
        yv += cmat[bc_base + Int32(3) * seqlen + l] * h4

        da = exp(dt * a[a_base + Int32(5)])
        dbv = dt * bmat[bc_base + Int32(4) * seqlen + l]
        h5 = da * h5 + dbv * uv
        yv += cmat[bc_base + Int32(4) * seqlen + l] * h5

        da = exp(dt * a[a_base + Int32(6)])
        dbv = dt * bmat[bc_base + Int32(5) * seqlen + l]
        h6 = da * h6 + dbv * uv
        yv += cmat[bc_base + Int32(5) * seqlen + l] * h6

        da = exp(dt * a[a_base + Int32(7)])
        dbv = dt * bmat[bc_base + Int32(6) * seqlen + l]
        h7 = da * h7 + dbv * uv
        yv += cmat[bc_base + Int32(6) * seqlen + l] * h7

        da = exp(dt * a[a_base + Int32(8)])
        dbv = dt * bmat[bc_base + Int32(7) * seqlen + l]
        h8 = da * h8 + dbv * uv
        yv += cmat[bc_base + Int32(7) * seqlen + l] * h8

        y[row_base + l] = yv * silu_f(z[row_base + l])
        l += Int32(1)
    end

    ssm_states[state_base + Int32(1)] = h1
    ssm_states[state_base + Int32(2)] = h2
    ssm_states[state_base + Int32(3)] = h3
    ssm_states[state_base + Int32(4)] = h4
    ssm_states[state_base + Int32(5)] = h5
    ssm_states[state_base + Int32(6)] = h6
    ssm_states[state_base + Int32(7)] = h7
    ssm_states[state_base + Int32(8)] = h8
    end
end

include(joinpath(@__DIR__,"..","study_common.jl"))
const PORTABLE=portable_scan!(CUDABackend())
function launch!(f,y)
 a=f[1];batch,dim,L=f[6:8];tx=min(Int(dim),1024)
 PORTABLE(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],f[2],y,batch,dim,Int32(8),L;ndrange=(cld(Int(dim),tx)*tx,Int(batch)),workgroupsize=(tx,1))
end
study_main("ka",launch!)

using CUDA
function permute_kernel!(q, k, v, inp, total::Int32, tdim::Int32, nh::Int32, d::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 < total
        c = nh * d
        b = idx0 ÷ (c * tdim)
        rest = idx0 % (c * tdim)
        nh_ = rest ÷ (tdim * d)
        rest = rest % (tdim * d)
        n = rest ÷ d
        d_ = rest % d
        inp_idx = b * tdim * Int32(3) * c + n * Int32(3) * c + nh_ * d + d_ + Int32(1)
        out_idx = idx0 + Int32(1)
        @inbounds begin
            q[out_idx] = inp[inp_idx]
            k[out_idx] = inp[inp_idx + c]
            v[out_idx] = inp[inp_idx + Int32(2) * c]
        end
    end
    return
end
function factory(out,inp,B,T,C,NH,bs)
 total=B*T*C;q=view(out,1:total);k=view(out,total+1:2total);v=view(out,2total+1:3total);d=C÷NH
 kern=@cuda launch=false permute_kernel!(q,k,v,inp,Int32(total),Int32(T),Int32(NH),Int32(d))
 ()->begin;kern(q,k,v,inp,Int32(total),Int32(T),Int32(NH),Int32(d);threads=bs,blocks=cld(total,bs));CUDA.device_synchronize(blocking=true);end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
permutation_study(factory)

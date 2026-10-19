using CUDA
include("../device_math.jl")
include("../study_common.jl")
function init1_cuda(nodes,a,u)
    f_init1!(nodes,FZero(a),u,threadIdx().x-Int32(1),blockIdx().x-Int32(1));return
end
function init2_cuda(nodes,idx,nei,wt,a,u)
    f_init2!(nodes,FZero(idx),FZero(nei),FZero(wt),FZero(a),u,threadIdx().x-Int32(1),blockIdx().x-Int32(1));return
end
function fw0_cuda(a,u,kr,kc)
    f_FW0_64!(FZero(a),u,FZero(kr),FZero(kc),threadIdx().x-Int32(1),blockIdx().x-Int32(1));return
end
function rowcol_cuda(a,u,kr,kc,x,sm)
    f_FWrowcol_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,threadIdx().x-Int32(1),blockIdx().x-Int32(1));return
end
function rem_cuda(a,u,kr,kc,x,sm)
    f_FWrem_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,threadIdx().x-Int32(1),blockIdx().x-Int32(1));return
end
function observer_cuda(out,n,e,u,r)
    if threadIdx().x==1
        @inbounds out[1]=n;out[2]=e;out[3]=u;out[4]=r
    end
    return
end
function init_cuda(nodes,idx,nei,wt,a,u)
    @cuda threads=1024 blocks=cld(Int(u)*Int(u),1024) maxregs=64 init1_cuda(nodes,a,u)
    @cuda threads=1024 blocks=cld(Int(nodes),1024) maxregs=64 init2_cuda(nodes,idx,nei,wt,a,u)
    nothing
end
function compute_cuda(a,u,kr,kc)
    sub=u÷Int32(64);sm=sub-Int32(1)
    @cuda threads=1024 blocks=1 maxregs=64 fw0_cuda(a,u,kr,kc)
    if sub>1
        for x in Int32(0):sub-Int32(1)
            @cuda threads=1024 blocks=2*Int(sm) maxregs=32 rowcol_cuda(a,u,kr,kc,x,sm)
            @cuda threads=1024 blocks=Int(sm)*Int(sm) maxregs=32 rem_cuda(a,u,kr,kc,x,sm)
        end
    end
    nothing
end
observe_cuda(out,n,e,u,r)=(@cuda threads=32 observer_cuda(out,n,e,u,r);nothing)
floyd_study(CuArray,init_cuda,compute_cuda,observe_cuda)

using CUDA, KernelAbstractions
include("../device_math.jl")
include("../study_common.jl")
@kernel function init1_ka!(nodes,a,u)
local_id=@index(Local,Linear)
group_id=@index(Group,Linear)
tid=Int32(local_id-1);block=Int32(group_id-1)
f_init1!(nodes,FZero(a),u,tid,block)
nothing
end
@kernel function init2_ka!(nodes,idx,nei,wt,a,u)
local_id=@index(Local,Linear)
group_id=@index(Group,Linear)
tid=Int32(local_id-1);block=Int32(group_id-1)
f_init2!(nodes,FZero(idx),FZero(nei),FZero(wt),FZero(a),u,tid,block)
nothing
end
@kernel function fw0_ka!(a,u,kr,kc)
local_id=@index(Local,Linear)
group_id=@index(Group,Linear)
tid=Int32(local_id-1);block=Int32(group_id-1)
f_FW0_64!(FZero(a),u,FZero(kr),FZero(kc),tid,block)
nothing
end
@kernel function rowcol_ka!(a,u,kr,kc,x,sm)
local_id=@index(Local,Linear)
group_id=@index(Group,Linear)
tid=Int32(local_id-1);block=Int32(group_id-1)
f_FWrowcol_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,tid,block)
nothing
end
@kernel function rem_ka!(a,u,kr,kc,x,sm)
local_id=@index(Local,Linear)
group_id=@index(Group,Linear)
tid=Int32(local_id-1);block=Int32(group_id-1)
f_FWrem_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,tid,block)
nothing
end
@kernel function observer_ka!(out,n,e,u,r)
    i=@index(Global,Linear)
    if i==1
        @inbounds out[1]=n;out[2]=e;out[3]=u;out[4]=r
    end
end
const fw_backend=CUDA.CUDABackend()
function allocate_ka(a)
    d=KernelAbstractions.allocate(fw_backend,eltype(a),size(a));copyto!(d,a);d
end
function initialize_ka(nodes,idx,nei,wt,a,u)
    init1_ka!(fw_backend,1024)(nodes,a,u;ndrange=cld(Int(u)*Int(u),1024)*1024)
    init2_ka!(fw_backend,1024)(nodes,idx,nei,wt,a,u;ndrange=cld(Int(nodes),1024)*1024)
    nothing
end
function compute_ka(a,u,kr,kc)
    sub=u÷Int32(64);sm=sub-Int32(1)
    fw0_ka!(fw_backend,1024)(a,u,kr,kc;ndrange=1024)
    if sub>1
        for x in Int32(0):sub-Int32(1)
            rowcol_ka!(fw_backend,1024)(a,u,kr,kc,x,sm;ndrange=2*Int(sm)*1024)
            rem_ka!(fw_backend,1024)(a,u,kr,kc,x,sm;ndrange=Int(sm)*Int(sm)*1024)
        end
    end
    nothing
end
observe_ka(out,n,e,u,r)=(observer_ka!(fw_backend,32)(out,n,e,u,r;ndrange=32);nothing)
floyd_study(allocate_ka,initialize_ka,compute_ka,observe_ka)


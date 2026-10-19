using CUDA
import JACC
JACC.@init_backend
include("../device_math.jl")
include("../study_common.jl")
function init1_jacc!(i,nodes,a,u)
tid=Int32((i-1)%1024);block=Int32((i-1)÷1024)
f_init1!(nodes,FZero(a),u,tid,block)
nothing
end
function init2_jacc!(i,nodes,idx,nei,wt,a,u)
tid=Int32((i-1)%1024);block=Int32((i-1)÷1024)
f_init2!(nodes,FZero(idx),FZero(nei),FZero(wt),FZero(a),u,tid,block)
nothing
end
function fw0_jacc!(i,a,u,kr,kc)
tid=Int32((i-1)%1024);block=Int32((i-1)÷1024)
f_FW0_64!(FZero(a),u,FZero(kr),FZero(kc),tid,block)
nothing
end
function rowcol_jacc!(i,a,u,kr,kc,x,sm)
tid=Int32((i-1)%1024);block=Int32((i-1)÷1024)
f_FWrowcol_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,tid,block)
nothing
end
function rem_jacc!(i,a,u,kr,kc,x,sm)
tid=Int32((i-1)%1024);block=Int32((i-1)÷1024)
f_FWrem_64!(FZero(a),u,FZero(kr),FZero(kc),x,sm,tid,block)
nothing
end
function observer_jacc!(i,out,n,e,u,r)
    if i==1
        @inbounds out[1]=n;out[2]=e;out[3]=u;out[4]=r
    end
    nothing
end
function fw_jlaunch(f::F,blocks,args...) where F
    spec=JACC.launch_spec(threads=1024,blocks=blocks,shmem_size=0,sync=false)
    JACC.parallel_for(spec,blocks*1024,f,args...)
    nothing
end
function initialize_jacc(nodes,idx,nei,wt,a,u)
    fw_jlaunch(init1_jacc!,cld(Int(u)*Int(u),1024),nodes,a,u)
    fw_jlaunch(init2_jacc!,cld(Int(nodes),1024),nodes,idx,nei,wt,a,u)
    nothing
end
function compute_jacc(a,u,kr,kc)
    sub=u÷Int32(64);sm=sub-Int32(1)
    fw_jlaunch(fw0_jacc!,1,a,u,kr,kc)
    if sub>1
        for x in Int32(0):sub-Int32(1)
            fw_jlaunch(rowcol_jacc!,2*Int(sm),a,u,kr,kc,x,sm)
            fw_jlaunch(rem_jacc!,Int(sm)*Int(sm),a,u,kr,kc,x,sm)
        end
    end
    nothing
end
function observe_jacc(out,n,e,u,r)
    spec=JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false)
    JACC.parallel_for(spec,32,observer_jacc!,out,n,e,u,r)
    nothing
end
floyd_study(JACC.array,initialize_jacc,compute_jacc,observe_jacc)


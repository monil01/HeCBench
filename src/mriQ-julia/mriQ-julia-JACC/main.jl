using CUDA
import JACC
JACC.@init_backend
function phi_jacc!(i,pr,pii,mag,k)
    if i<=k
        @inbounds mag[i]=pr[i]*pr[i]+pii[i]*pii[i]
    end
    nothing
end
function q_jacc!(j,tile,k,base,x,y,z,qr,qi)
    @inbounds begin
        sx=x[j];sy=y[j];sz=z[j];rr=qr[j];ii=qi[j]
        index=Int32(0);globalindex=base
        if isodd(k)
            angle=6.2831855f0*(tile[1]*sx+tile[2]*sy+tile[3]*sz)
            rr+=tile[4]*cos(angle);ii+=tile[4]*sin(angle)
            index+=Int32(1);globalindex+=Int32(1)
        end
        while index<1024 && globalindex<k
            offset=4index
            angle=6.2831855f0*(tile[offset+1]*sx+tile[offset+2]*sy+tile[offset+3]*sz)
            rr+=tile[offset+4]*cos(angle);ii+=tile[offset+4]*sin(angle)
            offset+=4
            angle=6.2831855f0*(tile[offset+1]*sx+tile[offset+2]*sy+tile[offset+3]*sz)
            rr+=tile[offset+4]*cos(angle);ii+=tile[offset+4]*sin(angle)
            index+=Int32(2);globalindex+=Int32(2)
        end
        qr[j]=rr;qi[j]=ii
    end
    nothing
end
launchphi!(r,i,m,k)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(Int(k),256),shmem_size=0,sync=false),Int(k),phi_jacc!,r,i,m,k)
launchq!(tile,k,base,x,y,z,qr,qi,n)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,q_jacc!,tile,k,base,x,y,z,qr,qi)
include(joinpath(@__DIR__,"..","study_common.jl"))
mriq_study(JACC.array,launchphi!,launchq!)

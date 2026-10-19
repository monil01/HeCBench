using CUDA
using KernelAbstractions
@kernel function phi_ka!(pr,pii,mag,k)
    i=@index(Global,Linear)
    if i<=k
        @inbounds mag[i]=pr[i]*pr[i]+pii[i]*pii[i]
    end
    nothing
end
@kernel function q_ka!(tile,k,base,x,y,z,qr,qi)
    j=@index(Global,Linear)
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
function ka_array(x)
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),length(x))
    copyto!(a,x)
    a
end
launchphi!(r,i,m,k)=phi_ka!(CUDABackend(),256)(r,i,m,k;ndrange=Int(k))
launchq!(tile,k,base,x,y,z,qr,qi,n)=q_ka!(CUDABackend(),256)(tile,k,base,x,y,z,qr,qi;ndrange=n)
include(joinpath(@__DIR__,"..","study_common.jl"))
mriq_study(ka_array,launchphi!,launchq!)

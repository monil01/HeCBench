using CUDA
function phi_cuda!(pr,pii,mag,k)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=k
        @inbounds mag[i]=pr[i]*pr[i]+pii[i]*pii[i]
    end
    nothing
end
function q_cuda!(tile,k,base,x,y,z,qr,qi)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
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
launchphi!(r,i,m,k)=@cuda threads=256 blocks=cld(k,256) phi_cuda!(r,i,m,k)
launchq!(tile,k,base,x,y,z,qr,qi,n)=@cuda threads=256 blocks=cld(n,256) q_cuda!(tile,k,base,x,y,z,qr,qi)
include(joinpath(@__DIR__,"..","study_common.jl"))
mriq_study(CuArray,launchphi!,launchq!)

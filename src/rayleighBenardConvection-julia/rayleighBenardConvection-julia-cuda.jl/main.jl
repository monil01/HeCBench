using CUDA,Libdl,SHA
@inline function rbc_index()
    ((Int(blockIdx().y)-1)*Int(gridDim().x)+Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)
end
function omega_cuda!(A,B,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function temperature_cuda!(A,B,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function heat_cuda!(A,B,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function subtract_one_cuda!(A,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]-1f0;end;nothing
end
function add_one_cuda!(A,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]+1f0;end;nothing
end
function add_x_cuda!(A,x,n)
    i=rbc_index();if i<=n;@inbounds A[i]=A[i]+x;end;nothing
end
function adaptive_dt_cuda!(u,v,iu,iv,out,dx,cap)
    @inbounds out[1]=min(dx/max(abs(u[iu]),abs(v[iv])),cap);nothing
end
function rbc_launch(q,A,B,out)
    threads=Int(q.threads);blocks=(Int(q.gx),Int(q.gy));n=Int(q.n);s=CUDA.legacy_stream()
    if q.op==0
        @cuda threads=threads blocks=blocks stream=s omega_cuda!(A,B,n)
    elseif q.op==1
        @cuda threads=threads blocks=blocks stream=s temperature_cuda!(A,B,n)
    elseif q.op==2
        @cuda threads=threads blocks=blocks stream=s heat_cuda!(A,B,n)
    elseif q.op==3
        @cuda threads=threads blocks=blocks stream=s subtract_one_cuda!(A,n)
    elseif q.op==4
        @cuda threads=threads blocks=blocks stream=s add_one_cuda!(A,n)
    elseif q.op==5
        @cuda threads=threads blocks=blocks stream=s add_x_cuda!(A,reinterpret(Float32,q.pad),n)
    elseif q.op==6
        @cuda threads=threads blocks=blocks stream=s adaptive_dt_cuda!(A,B,Int(q.iu),Int(q.iv),out,RBC_DX,RBC_CAP)
    else
        error("Unknown source pointwise operation")
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rbc_study()

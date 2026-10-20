using CUDA,KernelAbstractions,Libdl,SHA
@kernel function omega_ka!(A,B,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]*B[i];end
end
@kernel function temperature_ka!(A,B,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]*B[i];end
end
@kernel function heat_ka!(A,B,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]*B[i];end
end
@kernel function subtract_one_ka!(A,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]-1f0;end
end
@kernel function add_one_ka!(A,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]+1f0;end
end
@kernel function add_x_ka!(A,x,n)
    i=@index(Global,Linear);if i<=n;@inbounds A[i]=A[i]+x;end
end
@kernel function adaptive_dt_ka!(u,v,iu,iv,out,dx,cap)
    i=@index(Global,Linear)
    if i==1;@inbounds out[1]=min(dx/max(abs(u[iu]),abs(v[iv])),cap);end
end
function rbc_launch(q,A,B,out)
    threads=Int(q.threads);range=threads*Int(q.gx)*Int(q.gy);n=Int(q.n)
    backend=KernelAbstractions.get_backend(A)
    if q.op==0
        omega_ka!(backend,threads)(A,B,n;ndrange=range)
    elseif q.op==1
        temperature_ka!(backend,threads)(A,B,n;ndrange=range)
    elseif q.op==2
        heat_ka!(backend,threads)(A,B,n;ndrange=range)
    elseif q.op==3
        subtract_one_ka!(backend,threads)(A,n;ndrange=range)
    elseif q.op==4
        add_one_ka!(backend,threads)(A,n;ndrange=range)
    elseif q.op==5
        add_x_ka!(backend,threads)(A,reinterpret(Float32,q.pad),n;ndrange=range)
    elseif q.op==6
        adaptive_dt_ka!(backend,threads)(A,B,Int(q.iu),Int(q.iv),out,RBC_DX,RBC_CAP;ndrange=range)
    else
        error("Unknown source pointwise operation")
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rbc_study()

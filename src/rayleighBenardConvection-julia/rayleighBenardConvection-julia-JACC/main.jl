using CUDA,JACC,Libdl,SHA
function omega_jacc!(i,A,B,n)
    if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function temperature_jacc!(i,A,B,n)
    if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function heat_jacc!(i,A,B,n)
    if i<=n;@inbounds A[i]=A[i]*B[i];end;nothing
end
function subtract_one_jacc!(i,A,n)
    if i<=n;@inbounds A[i]=A[i]-1f0;end;nothing
end
function add_one_jacc!(i,A,n)
    if i<=n;@inbounds A[i]=A[i]+1f0;end;nothing
end
function add_x_jacc!(i,A,x,n)
    if i<=n;@inbounds A[i]=A[i]+x;end;nothing
end
function adaptive_dt_jacc!(i,u,v,iu,iv,out,dx,cap)
    if i==1;@inbounds out[1]=min(dx/max(abs(u[iu]),abs(v[iv])),cap);end;nothing
end
const RBC_JACC_SPECS=Dict{Tuple{Int,Int},Any}()
function rbc_spec(threads,blocks)
    get!(RBC_JACC_SPECS,(threads,blocks)) do
        JACC.launch_spec(;threads=threads,blocks=blocks,shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
end
function rbc_launch(q,A,B,out)
    threads=Int(q.threads);blocks=Int(q.gx)*Int(q.gy);range=threads*blocks;n=Int(q.n);spec=rbc_spec(threads,blocks)
    if q.op==0
        JACC.parallel_for(spec,range,omega_jacc!,A,B,n)
    elseif q.op==1
        JACC.parallel_for(spec,range,temperature_jacc!,A,B,n)
    elseif q.op==2
        JACC.parallel_for(spec,range,heat_jacc!,A,B,n)
    elseif q.op==3
        JACC.parallel_for(spec,range,subtract_one_jacc!,A,n)
    elseif q.op==4
        JACC.parallel_for(spec,range,add_one_jacc!,A,n)
    elseif q.op==5
        JACC.parallel_for(spec,range,add_x_jacc!,A,reinterpret(Float32,q.pad),n)
    elseif q.op==6
        JACC.parallel_for(spec,range,adaptive_dt_jacc!,A,B,Int(q.iu),Int(q.iv),out,RBC_DX,RBC_CAP)
    else
        error("Unknown source pointwise operation")
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rbc_study()

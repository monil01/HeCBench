using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function distance_jacc!(ix,iy,A,B,sa,sb,partial,m::Int32,n::Int32,template)
 T=eltype(A);x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1);tx=x%Int32(32);ty=y%Int32(8);bx=x÷Int32(32);by=y÷Int32(8);s=JACC.shared(template)
 if tx==0&&ty==0;@inbounds s[481]=zero(T);end
 exp_fill!(s,A,B,sa,sb,m,n,tx,ty,bx,by);JACC.sync_workgroup()
 v=exp_sum(s,tx,ty,T);JACC.@atomic s[481]+=v;JACC.sync_workgroup()
 if tx==0&&ty==0;@inbounds partial[by*(m÷Int32(128))+bx+Int32(1)]=s[481];end
 return
end
function reduce_jacc!(index,out,partial,nb::Int32,template)
 T=eltype(out);tx=Int32(index)-Int32(1);s=JACC.shared(template)
 if tx==0;@inbounds s[1]=zero(T);end
 JACC.sync_workgroup();v=zero(T);i=tx
 while i<nb;@inbounds v+=partial[i+Int32(1)];i+=Int32(256);end
 JACC.@atomic s[1]+=v;JACC.sync_workgroup()
 if tx==0;@inbounds out[1]=s[1];end
 return
end
function factory(A,B,sa,sb,partial,out,m,n)
 T=eltype(A);template=CUDA.zeros(T,481);rt=CUDA.zeros(T,1);gx=m÷128;gy=n÷32;nb=gx*gy
 spec=JACC.launch_spec(;threads=(32,8),blocks=(gx,gy),shmem_size=481*sizeof(T),sync=false);rs=JACC.launch_spec(;threads=256,blocks=1,shmem_size=sizeof(T),sync=false)
 ()->begin;JACC.parallel_for(spec,(m÷4,n÷4),distance_jacc!,A,B,sa,sb,partial,Int32(m),Int32(n),template);JACC.parallel_for(rs,256,reduce_jacc!,out,partial,Int32(nb),rt);end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
exp_study(factory)

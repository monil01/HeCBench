using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function distance_ka!(A::AbstractArray{T},B,sa,sb,partial,m::Int32,n::Int32) where T
 localid=@index(Local,NTuple);groupid=@index(Group,NTuple);tx=Int32(localid[1])-Int32(1);ty=Int32(localid[2])-Int32(1);bx=Int32(groupid[1])-Int32(1);by=Int32(groupid[2])-Int32(1);s=@localmem T (481,)
 if tx==0&&ty==0;@inbounds s[481]=zero(T);end
 exp_fill!(s,A,B,sa,sb,m,n,tx,ty,bx,by)
 @synchronize
 v=exp_sum(s,tx,ty,T);Atomix.@atomic s[481]+=v
 @synchronize
 if tx==0&&ty==0;@inbounds partial[by*(m÷Int32(128))+bx+Int32(1)]=s[481];end
end
@kernel function reduce_ka!(out::AbstractArray{T},partial,nb::Int32) where T
 tx=Int32(@index(Local,Linear))-Int32(1);s=@localmem T (1,)
 if tx==0;@inbounds s[1]=zero(T);end
 @synchronize
 v=zero(T);i=tx
 while i<nb;@inbounds v+=partial[i+Int32(1)];i+=Int32(256);end
 Atomix.@atomic s[1]+=v
 @synchronize
 if tx==0;@inbounds out[1]=s[1];end
end
function factory(A,B,sa,sb,partial,out,m,n)
 k=distance_ka!(CUDABackend(),(32,8));r=reduce_ka!(CUDABackend(),256);nb=(m÷128)*(n÷32)
 ()->begin;k(A,B,sa,sb,partial,Int32(m),Int32(n);ndrange=(m÷4,n÷4));r(out,partial,Int32(nb);ndrange=256);end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
exp_study(factory)

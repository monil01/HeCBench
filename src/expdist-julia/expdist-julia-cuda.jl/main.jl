using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function distance_jl!(A,B,sa,sb,partial,m::Int32,n::Int32)
 T=eltype(A);tx=threadIdx().x-Int32(1);ty=threadIdx().y-Int32(1);bx=blockIdx().x-Int32(1);by=blockIdx().y-Int32(1);s=CuStaticSharedArray(T,481)
 if tx==0&&ty==0;@inbounds s[481]=zero(T);end
 exp_fill!(s,A,B,sa,sb,m,n,tx,ty,bx,by);sync_threads()
 v=exp_sum(s,tx,ty,T);CUDA.@atomic s[481]+=v;sync_threads()
 if tx==0&&ty==0;@inbounds partial[by*gridDim().x+bx+Int32(1)]=s[481];end
 return
end
function reduce_jl!(out,partial,nb::Int32)
 T=eltype(out);tx=threadIdx().x-Int32(1);s=CuStaticSharedArray(T,1)
 if tx==0;@inbounds s[1]=zero(T);end
 sync_threads();v=zero(T);i=tx
 while i<nb;@inbounds v+=partial[i+Int32(1)];i+=Int32(256);end
 CUDA.@atomic s[1]+=v;sync_threads()
 if tx==0;@inbounds out[1]=s[1];end
 return
end
function factory(A,B,sa,sb,partial,out,m,n)
 gx=m÷128;gy=n÷32;nb=gx*gy
 k=@cuda launch=false distance_jl!(A,B,sa,sb,partial,Int32(m),Int32(n));r=@cuda launch=false reduce_jl!(out,partial,Int32(nb))
 ()->begin;k(A,B,sa,sb,partial,Int32(m),Int32(n);threads=(32,8),blocks=(gx,gy));r(out,partial,Int32(nb);threads=256,blocks=1);end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
exp_study(factory)

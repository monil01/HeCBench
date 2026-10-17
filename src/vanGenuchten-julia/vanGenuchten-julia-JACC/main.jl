using CUDA
import JACC
JACC.@init_backend
function kernel!(i,Ksat,psi,C,theta,K,size)
 if i<=size
        a=0.02; ts=0.45; tr=0.1; nn=1.8; m=(nn-1.0)/nn
        @inbounds h=psi[i]*100.0
        th=h<0.0 ? (ts-tr)/(1.0+(a*(-h))^nn)^m+tr : ts
        se=(th-tr)/(ts-tr)
        t=1.0-(1.0-se^(1.0/m))^m
        @inbounds theta[i]=th
        @inbounds K[i]=Ksat[i]*sqrt(se)*t*t
        @inbounds C[i]=h<0.0 ? 100.0*a*nn*(1.0/nn-1.0)*(a*abs(h))^(nn-1.0)*(tr-ts)*((a*abs(h))^nn+1.0)^(1.0/nn-2.0) : 0.0
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
run!(s,p,c,t,k,n)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,kernel!,s,p,c,t,k,Int32(n));nothing)
study(run!)

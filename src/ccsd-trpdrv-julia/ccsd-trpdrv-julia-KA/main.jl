using KernelAbstractions,CUDA,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function ccsd_ka!(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,ea,out,nv)
 b,c=@index(Global,NTuple)
 if b<=nv && c<=nv
  s1,s3,s2,s4=ccsd_terms(b,c,f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,ea,Int32(0),Int32(3),nv)
  Atomix.@atomic out[1]+=s1
  Atomix.@atomic out[2]+=s3
  Atomix.@atomic out[3]+=s2
  Atomix.@atomic out[4]+=s4
 end
end
function study_launch!(a,eo,ea,o,nv)
 ccsd_ka!(CUDABackend(),(16,16))(a...,eo,ea,o,Int32(nv);ndrange=(cld(nv,16)*16,cld(nv,16)*16))
end
study_main(study_launch!)

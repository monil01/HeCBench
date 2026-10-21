using CUDA,JACC,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
const CCSD_BACKEND=JACC.get_backend(:cuda)
const CCSD_SPEC=JACC.LaunchSpec{typeof(CCSD_BACKEND)}(;threads=(16,16),blocks=(cld(CV,16),cld(CV,16)),shmem_size=0,sync=false)
function ccsd_jacc!(b,c,f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,ea,out,nv)
 s1,s3,s2,s4=ccsd_terms(b,c,f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,ea,Int32(0),Int32(3),nv)
 Atomix.@atomic out[1]+=s1
 Atomix.@atomic out[2]+=s3
 Atomix.@atomic out[3]+=s2
 Atomix.@atomic out[4]+=s4
 nothing
end
function study_launch!(a,eo,ea,o,nv)
 JACC.parallel_for(ccsd_jacc!,CCSD_SPEC,(nv,nv),a...,eo,ea,o,Int32(nv))
end
study_main(study_launch!)

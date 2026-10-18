using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
function aos_jacc!(i,A,R,P,n,M)
 private_body!(Int32(i-1),A,R,P,n,M,false);return
end
function soa_jacc!(i,A,R,P,n,M)
 private_body!(Int32(i-1),A,R,P,n,M,true);return
end
function shared_jacc!(i,A,R,P,n,M,template)
 s=JACC.shared(template);row=Int32(i-1);tid=row%Int32(96);shared_body!(row,tid,s,A,R,P,n,M);return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATE=JACC.zeros(Float64,4032)
function launch(variant,A,R,P,n,M)
 if variant==3
  blocks=cld(Int(n),96);JACC.parallel_for(JACC.launch_spec(threads=96,blocks=blocks,shmem_size=32256,sync=false),blocks*96,shared_jacc!,A,R,P,n,M,TEMPLATE)
 else
  blocks=cld(Int(n),192);f=variant==1 ? aos_jacc! : soa_jacc!;JACC.parallel_for(JACC.launch_spec(threads=192,blocks=blocks,shmem_size=0,sync=false),blocks*192,f,A,R,P,n,M)
 end
 return
end
study(launch,(T,n)->JACC.zeros(T,n))

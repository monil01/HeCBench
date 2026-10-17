using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=JACC.get_backend(:cuda)
const SPECS=Dict((f[1],f[2],f[3])=>JACC.LaunchSpec{typeof(BACKEND)}(threads=(64,1,1),blocks=(f[2],f[1],cld(f[3],64)),shmem_size=0,sync=false) for f in fixtures())
@inline function metadata_jacc!(x,y,z,q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 h=Int64((x-1)÷64);b=Int64(y-1);row=Int64(z-1)*64+Int64((x-1)%64)
 metadata_row!(row,h,b,q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 return
end
function launch!(p)
 f,ins,outs,causal=p;B,H,R,M,BN,V,S=f[1:7]
 JACC.parallel_for(metadata_jacc!,SPECS[(B,H,R)],(64*H,B,cld(R,64)),ins...,outs...,H,R,M,BN,V,S,causal)
end
study_main(launch!)

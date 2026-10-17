using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function metadata_ka!(q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 group=@index(Group,NTuple);lane=@index(Local,Linear)
 h=Int64(group[1]-1);b=Int64(group[2]-1);row=Int64(group[3]-1)*64+Int64(lane-1)
 metadata_row!(row,h,b,q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
end
const META=metadata_ka!(CUDABackend(),(64,1,1))
function launch!(p)
 f,ins,outs,causal=p;B,H,R,M,BN,V,S=f[1:7]
 META(ins...,outs...,H,R,M,BN,V,S,causal;ndrange=(64*H,B,cld(R,64)))
end
study_main(launch!)

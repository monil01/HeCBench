using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATES=Dict(n=>JACC.zeros(Float32,n) for n in(96,551,560))
function launch(a,out,pars,bc,mode,slots)
 H,W,Ho,Wo,pitch,hs,he,ws,we=pars;nd=(cld(Int(Wo),16)*16,cld(Int(Ho),16)*16,bc);grid=(cld(Int(Wo),16),cld(Int(Ho),16),bc);spec=JACC.launch_spec(threads=(16,16,1),blocks=grid,shmem_size=4slots,sync=false)
 if mode==1;JACC.parallel_for(spec,nd,zoom_in_jacc!,a,out,pars...,TEMPLATES[slots])
 else;JACC.parallel_for(spec,nd,zoom_out_jacc!,a,out,pars...,TEMPLATES[slots]);JACC.parallel_for(JACC.launch_spec(threads=(16,16,1),blocks=(cld(Int(W),16),cld(Int(H),16),bc),shmem_size=0,sync=false),(cld(Int(W),16)*16,cld(Int(H),16)*16,bc),zoom_edge_jacc!,out,H,W,pitch,hs,ws,hs+Ho,ws+Wo)
 end
end
study((T,n)->JACC.zeros(T,n),launch)

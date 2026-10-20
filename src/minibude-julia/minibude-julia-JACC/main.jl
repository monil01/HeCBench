using CUDA
import JACC
JACC.@init_backend
include("../device_common.jl")
allocate_model(a)=JACC.array(a)
function fasten_jacc!(index,protein,ligand,ff,t0,t1,t2,t3,t4,t5,out,nt,n,nl,np)
    baseid=Int32(index)-Int32(1)
    ix=ifelse(baseid<n,baseid,n-Int32(1))
    T=prepare_transform(t0,t1,t2,t3,t4,t5,ix)
    cache=JACC.shared(view(ff,17:16+Int(nt)))
    evaluate_pose!(protein,ligand,cache,out,T,n,nl,np,baseid)
    nothing
end
function launch_bude(w)
    spec=JACC.launch_spec(threads=256,blocks=cld(w.n,256),shmem_size=16*w.nt,sync=false,stream=CUDA.legacy_stream())
    JACC.parallel_for(spec,256*cld(w.n,256),fasten_jacc!,w.protein,w.ligand,w.ff,w.poses...,w.out,Int32(w.nt),Int32(w.n),Int32(w.nl),Int32(w.np))
    nothing
end
include("../study_common.jl")
study_main("jacc")

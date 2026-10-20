using CUDA
include("../device_common.jl")
allocate_model(a)=CuArray(a)
function fasten_kernel!(protein,ligand,ff,t0,t1,t2,t3,t4,t5,out,nt,n,nl,np)
    tx=Int32(threadIdx().x)-Int32(1)
    baseid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tx
    ix=ifelse(baseid<n,baseid,n-Int32(1))
    cache=CuDynamicSharedArray(BudeFF,Int(nt))
    i=tx
    @inbounds while i<nt
        cache[i+Int32(1)]=ff[i+Int32(17)]
        i+=Int32(blockDim().x)
    end
    T=prepare_transform(t0,t1,t2,t3,t4,t5,ix)
    sync_threads()
    evaluate_pose!(protein,ligand,cache,out,T,n,nl,np,baseid)
    return nothing
end
function launch_bude(w)
    @cuda threads=256 blocks=cld(w.n,256) shmem=w.nt*16 fasten_kernel!(w.protein,w.ligand,w.ff,w.poses...,w.out,Int32(w.nt),Int32(w.n),Int32(w.nl),Int32(w.np))
    nothing
end
include("../study_common.jl")
study_main("cuda_jl")

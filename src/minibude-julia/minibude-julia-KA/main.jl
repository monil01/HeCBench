using CUDA, KernelAbstractions
include("../device_common.jl")
allocate_model(a)=KernelAbstractions.allocate(CUDABackend(),eltype(a),size(a)) |> d->(copyto!(d,a);d)
@kernel function fasten_ka!(protein,ligand,ff,t0,t1,t2,t3,t4,t5,out,::Val{NT},n,nl,np) where NT
    tx=Int32(@index(Local,Linear))-Int32(1)
    baseid=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+tx
    ix=ifelse(baseid<n,baseid,n-Int32(1))
    cache=@localmem BudeFF (NT,)
    i=tx
    @inbounds while i<Int32(NT)
        cache[i+Int32(1)]=ff[i+Int32(17)]
        i+=Int32(256)
    end
    T=prepare_transform(t0,t1,t2,t3,t4,t5,ix)
    @synchronize
    evaluate_pose!(protein,ligand,cache,out,T,n,nl,np,baseid)
end
function launch_bude(w)
    fasten_ka!(get_backend(w.out),256)(w.protein,w.ligand,w.ff,w.poses...,w.out,Val(w.nt),Int32(w.n),Int32(w.nl),Int32(w.np);ndrange=256*cld(w.n,256))
    nothing
end
include("../study_common.jl")
study_main("ka")

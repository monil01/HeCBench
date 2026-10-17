using CUDA
using JACC
include(joinpath(@__DIR__,"..","adv_stages.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function adv_kernel!(index,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
    idx=Int32(index)-Int32(1)
    adv_stage!(idx,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
end
function build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E)
    backend=JACC.get_backend(:cuda)
    stages=[(stage,n,JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)) for (stage,n) in ((Val(1),3*E*1024),(Val(2),3*E*2048),(Val(3),3*E*4096),(Val(4),3*E*4096),(Val(5),3*E*512),(Val(6),3*E*512),(Val(7),3*E*512))]
    ()->begin
        for (stage,n,spec) in stages
            JACC.parallel_for(adv_kernel!,spec,n,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
        end
    end
end
function study_prepare(vgeo,cg,D,I,u,out,E)
    r1,r2,cu,flux,tp,sp=adv_workspaces(E)
    build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,Int32(E))
end
exit(study_main(study_prepare))

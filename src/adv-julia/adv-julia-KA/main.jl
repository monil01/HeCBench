using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","adv_stages.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function adv_kernel!(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
    idx=Int32(@index(Global,Linear))-Int32(1)
    adv_stage!(idx,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
end
function build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E)
    k=adv_kernel!(CUDABackend(),256)
    ()->begin
        for (stage,n) in ((Val(1),3*E*1024),(Val(2),3*E*2048),(Val(3),3*E*4096),(Val(4),3*E*4096),(Val(5),3*E*512),(Val(6),3*E*512),(Val(7),3*E*512))
            k(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage;ndrange=n)
        end
    end
end
function study_prepare(vgeo,cg,D,I,u,out,E)
    r1,r2,cu,flux,tp,sp=adv_workspaces(E)
    build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,Int32(E))
end
exit(study_main(study_prepare))

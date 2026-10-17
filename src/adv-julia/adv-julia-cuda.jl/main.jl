using CUDA
include(joinpath(@__DIR__,"..","adv_stages.jl"))
function adv_kernel!(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
    idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
    adv_stage!(idx,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
end
function build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E)
    ()->begin
        for (stage,n) in ((Val(1),3*E*1024),(Val(2),3*E*2048),(Val(3),3*E*4096),(Val(4),3*E*4096),(Val(5),3*E*512),(Val(6),3*E*512),(Val(7),3*E*512))
            @cuda threads=256 blocks=cld(n,256) adv_kernel!(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,stage)
        end
    end
end
function study_prepare(vgeo,cg,D,I,u,out,E)
    r1,r2,cu,flux,tp,sp=adv_workspaces(E)
    build(vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,Int32(E))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end

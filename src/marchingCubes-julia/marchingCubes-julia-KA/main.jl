using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel1(out)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 s=@localmem Float32 (64,)
 mc_stage1(tid,group,out,s)
end
@kernel function kernel2(mm,ids,count,iso)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 s=@localmem UInt32 (32,)
 mc_stage2(tid,group,mm,ids,count,s,iso)
end
@kernel function kernel3(ids,mm)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 mc_stage3(tid,group,ids,mm)
end
@kernel function kernel4(mm,ids1,ids2,count,n1,iso)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 s=@localmem UInt32 (32,)
 mc_stage4(tid,group,mm,ids1,ids2,count,n1,s,iso)
end
@kernel function kernel5(ids,dist,tri,edges,counts,checksum,coords,iso)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 s=@localmem UInt32 (353,)
 mc_stage5(tid,group,ids,dist,tri,edges,counts,checksum,coords,s,iso)
end
function factory(mm1,ids1,mm2,ids2,n1,n2,counts,checksum,coords,d,t,e,iso)
 backend=KernelAbstractions.get_backend(mm1);k1=kernel1(backend,256);k2=kernel2(backend,128);k3=kernel3(backend,400);k4=kernel4(backend,1024);k5=kernel5(backend,128)
 (()->k1(mm1;ndrange=73984*256),()->k2(mm1,ids1,n1,iso;ndrange=578*128),n->k3(ids1,mm2;ndrange=n*400),n->k4(mm2,ids1,ids2,n2,Int32(n),iso;ndrange=cld(n*225,1024)*1024),n->(()->k5(ids2,d,t,e,counts,checksum,coords,iso;ndrange=n*128)))
end
if abspath(PROGRAM_FILE)==@__FILE__;marching_study(factory);end

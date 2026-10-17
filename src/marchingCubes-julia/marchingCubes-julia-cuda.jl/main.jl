using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel1(out)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 s=CUDA.@cuStaticSharedMem(Float32,64)
 mc_stage1(tid,group,out,s);return
end
function kernel2(mm,ids,count,iso)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 s=CUDA.@cuStaticSharedMem(UInt32,32)
 mc_stage2(tid,group,mm,ids,count,s,iso);return
end
function kernel3(ids,mm)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 mc_stage3(tid,group,ids,mm);return
end
function kernel4(mm,ids1,ids2,count,n1,iso)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 s=CUDA.@cuStaticSharedMem(UInt32,32)
 mc_stage4(tid,group,mm,ids1,ids2,count,n1,s,iso);return
end
function kernel5(ids,dist,tri,edges,counts,checksum,coords,iso)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 s=CUDA.@cuStaticSharedMem(UInt32,353)
 mc_stage5(tid,group,ids,dist,tri,edges,counts,checksum,coords,s,iso);return
end
function factory(mm1,ids1,mm2,ids2,n1,n2,counts,checksum,coords,d,t,e,iso)
 (()->(@cuda threads=256 blocks=73984 kernel1(mm1)),()->(@cuda threads=128 blocks=578 kernel2(mm1,ids1,n1,iso)),n->(@cuda threads=400 blocks=n kernel3(ids1,mm2)),n->(@cuda threads=1024 blocks=cld(n*225,1024) kernel4(mm2,ids1,ids2,n2,Int32(n),iso)),n->(()->(@cuda threads=128 blocks=n kernel5(ids2,d,t,e,counts,checksum,coords,iso))))
end
if abspath(PROGRAM_FILE)==@__FILE__;marching_study(factory);end

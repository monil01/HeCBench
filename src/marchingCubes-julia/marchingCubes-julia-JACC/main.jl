using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel1(index,out,template)
 tid=Int32((index-1)%256);group=Int32((index-1)÷256)
 s=JACC.shared(template)
 mc_stage1(tid,group,out,s);return
end
function kernel2(index,mm,ids,count,iso,template)
 tid=Int32((index-1)%128);group=Int32((index-1)÷128)
 s=JACC.shared(template)
 mc_stage2(tid,group,mm,ids,count,s,iso);return
end
function kernel3(index,ids,mm)
 tid=Int32((index-1)%400);group=Int32((index-1)÷400)
 mc_stage3(tid,group,ids,mm);return
end
function kernel4(index,mm,ids1,ids2,count,n1,iso,template)
 tid=Int32((index-1)%1024);group=Int32((index-1)÷1024)
 s=JACC.shared(template)
 mc_stage4(tid,group,mm,ids1,ids2,count,n1,s,iso);return
end
function kernel5(index,ids,dist,tri,edges,counts,checksum,coords,iso,template)
 tid=Int32((index-1)%128);group=Int32((index-1)÷128)
 s=JACC.shared(template)
 mc_stage5(tid,group,ids,dist,tri,edges,counts,checksum,coords,s,iso);return
end
function factory(mm1,ids1,mm2,ids2,n1,n2,counts,checksum,coords,d,t,e,iso)
 tmp1=CUDA.zeros(Float32,64);tmp2=CUDA.zeros(UInt32,32);tmp5=CUDA.zeros(UInt32,353)
 s1=JACC.launch_spec(;threads=256,blocks=73984,shmem_size=256,sync=false);s2=JACC.launch_spec(;threads=128,blocks=578,shmem_size=128,sync=false)
 op1=()->JACC.parallel_for(s1,73984*256,kernel1,mm1,tmp1)
 op2=()->JACC.parallel_for(s2,578*128,kernel2,mm1,ids1,n1,iso,tmp2)
 op3=n->JACC.parallel_for(JACC.launch_spec(;threads=400,blocks=n,shmem_size=0,sync=false),n*400,kernel3,ids1,mm2)
 op4=n->JACC.parallel_for(JACC.launch_spec(;threads=1024,blocks=cld(n*225,1024),shmem_size=128,sync=false),cld(n*225,1024)*1024,kernel4,mm2,ids1,ids2,n2,Int32(n),iso,tmp2)
 op5=n->begin
  spec=JACC.launch_spec(;threads=128,blocks=n,shmem_size=1412,sync=false)
  ()->JACC.parallel_for(spec,n*128,kernel5,ids2,d,t,e,counts,checksum,coords,iso,tmp5)
 end
 (op1,op2,op3,op4,op5)
end
if abspath(PROGRAM_FILE)==@__FILE__;marching_study(factory);end

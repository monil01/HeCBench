using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","convolution_math.jl"))
@kernel unsafe_indices=true function bt_kernel!(t,a,c)
 x=Int32(@index(Group,NTuple)[1]-1);y=Int32(@index(Group,NTuple)[2]-1);part=Int32(@index(Group,NTuple)[3]-1);tx=Int32(@index(Local,NTuple)[1]-1);ty=Int32(@index(Local,NTuple)[2]-1)
 s=@localmem(Float32,(4608,))
 st_bt_load!(s,a,x,y,0,tx,ty,part,c);@synchronize()
 v=st_bt_first(s,tx,ty);@synchronize();st_bt_store!(s,v,tx,ty);@synchronize()
 v2=st_bt_second(s,tx,ty);@synchronize();st_bt_output!(t,v2,x,y,part,tx,ty,c)

end
@kernel unsafe_indices=true function outer_kernel!(o,a,w,c,::Val{NS}) where NS
 plane=Int32(@index(Group,NTuple)[1]-1);part=Int32(@index(Group,NTuple)[2]-1);tx=Int32(@index(Local,NTuple)[1]-1);ty=Int32(@index(Local,NTuple)[2]-1)
 s=@localmem(Float32,(NS,))
 st_outer_load!(s,a,plane,part,tx,ty,c);v=0f0;v2=0f0
 for k in Int32(0):c÷32-1
  st_outer_weight!(s,w,plane,k,tx,ty,c);@synchronize()
  v+=st_outer_chunk(s,k,tx,ty,c,false)
  if c==256;v2+=st_outer_chunk(s,k,tx,ty,c,true);end
  @synchronize()
 end
 st_outer_output!(o,v,v2,plane,part,tx,ty,c)

end
@kernel unsafe_indices=true function at_kernel!(o,ip,bias,scale,c)
 x=Int32(@index(Group,NTuple)[1]-1);y=Int32(@index(Group,NTuple)[2]-1);z=Int32(@index(Group,NTuple)[3]-1);tx=Int32(@index(Local,NTuple)[1]-1);ty=Int32(@index(Local,NTuple)[2]-1)
 s=@localmem(Float32,(36,));st_at_load!(s,ip,x,y,z,tx,ty,c);@synchronize()
 v=st_at_first(s,tx,ty);@synchronize();s[tx*6+ty+1]=v;@synchronize()
 st_at_output!(o,s,bias,scale,x,y,z,tx,ty,c)

end
@kernel unsafe_indices=true function one_kernel!(o,a,w,bias,scale,ci,co,oc,chunk,relu,::Val{NS}) where NS
 tile=Int32(@index(Group,NTuple)[1]-1);part=Int32(@index(Group,NTuple)[2]-1);tx=Int32(@index(Local,NTuple)[1]-1);ty=Int32(@index(Local,NTuple)[2]-1)
 s=@localmem(Float32,(NS,))
 st_one_load!(s,a,bias,scale,tile,part,tx,ty,ci,co,oc);@synchronize()
 for k in Int32(0):chunk:ci-1
  st_one_weight!(s,w,k,part,tx,ty,ci,co,oc,chunk);@synchronize()
  st_one_acc!(s,k,tx,ty,ci,oc,chunk);@synchronize()
 end
 st_one_output!(o,s,tile,part,tx,ty,ci,co,oc,chunk,relu)

end
const BACKEND=KernelAbstractions.CUDABackend()
function launch!(f)
 a,w,b,s,t,ip,o,ci,co,mode=f
 if mode<2
  bt_kernel!(BACKEND,(128,6,1))(t,a,ci;ndrange=(512,24,Int(ci)÷128))
  outer_kernel!(BACKEND,(Int(ci),ci==128 ? 8 : 4,1))(ip,t,w,ci,Val(48*Int(ci));ndrange=(36*Int(ci),2*(ci==128 ? 8 : 4),1))
  at_kernel!(BACKEND,(6,6,1))(o,ip,b,s,ci;ndrange=(24,24,Int(ci)))
 else
  oc=Int32(min(ci,co));chunk=Int32(ci==1024 ? 16 : ci==256 ? 32 : 64);ns=4*ci+chunk*oc+6*oc
  one_kernel!(BACKEND,(Int(oc),4,1))(o,a,w,b,s,ci,co,oc,chunk,mode==2||mode==4,Val(Int(ns));ndrange=(49*Int(oc),4*Int(co÷oc),1))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("ka",launch!)

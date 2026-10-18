using CUDA,JACC
include(joinpath(@__DIR__,"..","convolution_math.jl"))
function bt_callback(i,j,k,t,a,c,template)
 ix=Int32(i)-1;iy=Int32(j)-1;x=ix÷128;y=iy÷6;part=Int32(k)-1;tx=ix%128;ty=iy%6
 s=JACC.shared(template)
 st_bt_load!(s,a,x,y,0,tx,ty,part,c);JACC.sync_workgroup()
 v=st_bt_first(s,tx,ty);JACC.sync_workgroup();st_bt_store!(s,v,tx,ty);JACC.sync_workgroup()
 v2=st_bt_second(s,tx,ty);JACC.sync_workgroup();st_bt_output!(t,v2,x,y,part,tx,ty,c)
 nothing
end
function outer_callback(i,j,k,o,a,w,c,template)
 ix=Int32(i)-1;iy=Int32(j)-1;nt=c==128 ? Int32(8) : Int32(4);plane=ix÷c;part=iy÷nt;tx=ix%c;ty=iy%nt
 s=JACC.shared(template)
 st_outer_load!(s,a,plane,part,tx,ty,c);v=0f0;v2=0f0
 for kk in Int32(0):c÷32-1
  st_outer_weight!(s,w,plane,kk,tx,ty,c);JACC.sync_workgroup()
  v+=st_outer_chunk(s,kk,tx,ty,c,false)
  if c==256;v2+=st_outer_chunk(s,kk,tx,ty,c,true);end
  JACC.sync_workgroup()
 end
 st_outer_output!(o,v,v2,plane,part,tx,ty,c)
 nothing
end
function at_callback(i,j,k,o,ip,bias,scale,c,template)
 ix=Int32(i)-1;iy=Int32(j)-1;x=ix÷6;y=iy÷6;z=Int32(k)-1;tx=ix%6;ty=iy%6
 s=JACC.shared(template);st_at_load!(s,ip,x,y,z,tx,ty,c);JACC.sync_workgroup()
 v=st_at_first(s,tx,ty);JACC.sync_workgroup();s[tx*6+ty+1]=v;JACC.sync_workgroup()
 st_at_output!(o,s,bias,scale,x,y,z,tx,ty,c)
 nothing
end
function one_callback(i,j,k,o,a,w,bias,scale,ci,co,oc,chunk,relu,template)
 ix=Int32(i)-1;iy=Int32(j)-1;tile=ix÷oc;part=iy÷4;tx=ix%oc;ty=iy%4
 s=JACC.shared(template)
 st_one_load!(s,a,bias,scale,tile,part,tx,ty,ci,co,oc);JACC.sync_workgroup()
 for kk in Int32(0):chunk:ci-1
  st_one_weight!(s,w,kk,part,tx,ty,ci,co,oc,chunk);JACC.sync_workgroup()
  st_one_acc!(s,kk,tx,ty,ci,oc,chunk);JACC.sync_workgroup()
 end
 st_one_output!(o,s,tile,part,tx,ty,ci,co,oc,chunk,relu)
 nothing
end
const BACKEND=JACC.get_backend(:cuda)
function spec(threads,blocks,bytes)
 JACC.LaunchSpec{typeof(BACKEND)}(;threads,blocks,shmem_size=bytes,sync=false)
end
const TEMPLATES=Dict(n=>CUDA.zeros(Float32,n) for n in (36,4608,6144,12288,11008,9472,9728,10752))
const SPECS=Dict{Int,Any}()
for (m,(ci,co)) in enumerate(((128,128),(256,256),(512,128),(128,512),(1024,256),(256,1024)))
 if m<=2
  SPECS[m-1]=(spec((128,6,1),(4,4,ci÷128),18432),spec((ci,ci==128 ? 8 : 4,1),(36,2,1),48*ci*4),spec((6,6,1),(4,4,ci),144))
 else
  oc=min(ci,co);chunk=ci==1024 ? 16 : ci==256 ? 32 : 64;ns=4*ci+chunk*oc+6*oc
  SPECS[m-1]=spec((oc,4,1),(49,co÷oc,1),ns*4)
 end
end
function launch!(f)
 a,w,b,s,t,ip,o,ci,co,mode=f;sp=SPECS[Int(mode)]
 if mode<2
  JACC.parallel_for(bt_callback,sp[1],(512,24,Int(ci)÷128),t,a,ci,TEMPLATES[4608])
  JACC.parallel_for(outer_callback,sp[2],(36*Int(ci),2*(ci==128 ? 8 : 4),1),ip,t,w,ci,TEMPLATES[48*Int(ci)])
  JACC.parallel_for(at_callback,sp[3],(24,24,Int(ci)),o,ip,b,s,ci,TEMPLATES[36])
 else
  oc=Int32(min(ci,co));chunk=Int32(ci==1024 ? 16 : ci==256 ? 32 : 64);ns=Int(4*ci+chunk*oc+6*oc)
  JACC.parallel_for(one_callback,sp,(49*Int(oc),4*Int(co÷oc),1),o,a,w,b,s,ci,co,oc,chunk,mode==2||mode==4,TEMPLATES[ns])
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("jacc",launch!)

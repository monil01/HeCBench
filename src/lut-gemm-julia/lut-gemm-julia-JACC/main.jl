using CUDA,JACC
include("../study_device.jl")
const T32=CUDA.zeros(Float16,1024);const T64=CUDA.zeros(Float16,2048)
function lut_kernel!(i,j,W,A,B,X,Y,M,K,nb,G,vt,vb,template)
 t=(i-1)%256;bx=(i-1)÷256;by=j-1;lut=JACC.shared(template)
 lut_build!(lut,X,t,by,vt);JACC.sync_workgroup();lut_finish!(lut,W,A,B,X,Y,M,K,nb,G,t,bx,by,vt,vb);nothing
end
function launch(c)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(256,1),blocks=(cld(c.M,2048),c.K÷c.T),shmem_size=c.T÷8*256*2,sync=false)
 JACC.parallel_for(lut_kernel!,spec,(cld(c.M,2048)*256,c.K÷c.T),c.vs...,c.y,c.M,c.K,c.nb,c.G,Val(c.T),Val(c.bi==1),c.T==32 ? T32 : T64);nothing
end
include("../study_common.jl")
lut_study(launch)

using CUDA,JACC
include("../reaction_device.jl")
function derivative!(i,j,axis,f,out,M,template,bx,by)
 lx=rem(Int32(i)-Int32(1),bx)+Int32(1);ly=rem(Int32(j)-Int32(1),by)+Int32(1)
 gx=div(Int32(i)-Int32(1),bx)+Int32(1);gy=div(Int32(j)-Int32(1),by)+Int32(1)
 shared=JACC.shared(template)
 index,slot,coordinate,step=pencil_position(axis,lx,ly,gx,gy,M)
 @inbounds shared[slot]=f[index]
 JACC.sync_workgroup()
 @inbounds out[index]=pencil_difference(shared,slot,coordinate,step,M)
 nothing
end
function point_kernel!(i,op,out,d,N,coef,dt,c1,c2)
 point_site!(op,out,d,i,N,coef,dt,c1,c2)
 nothing
end
function diff_launch(axis::Val{A},f,out,c) where A
 M=Int(c.M);threads=A==1 ? (M,2) : (2,M);grid=(M÷2,M)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=threads,blocks=grid,shmem_size=2*M*sizeof(Float32),sync=false)
 JACC.parallel_for(derivative!,spec,(grid[1]*threads[1],grid[2]*threads[2]),axis,f,out,c.M,c.template,Int32(threads[1]),Int32(threads[2]))
 nothing
end
function point_launch(op,out,c,coef)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=Int(c.M),blocks=Int(c.N÷c.M),shmem_size=0,sync=false)
 JACC.parallel_for(point_kernel!,spec,Int(c.N),op,out,c.d,c.N,coef,c.dt,c.c1,c.c2)
 nothing
end
include("../study_common.jl")
reaction_study(diff_launch,point_launch)

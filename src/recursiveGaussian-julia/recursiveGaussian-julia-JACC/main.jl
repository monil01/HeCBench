
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function recursive_kernel!(i,input,out,w,h,gp)
 x=Int32(i-1)
 recursive_body!(x,view(input,17:length(input)-16),view(out,17:length(out)-16),w,h,gp);nothing
end
function transpose_kernel!(i,j,input,out,w,h,template)
 tx=Int32((i-1)%16);ty=Int32((j-1)%16);bx=Int32((i-1)÷16);by=Int32((j-1)÷16);tile=JACC.shared(template);x=bx*Int32(16)+tx;y=by*Int32(16)+ty
 if x<w&&y<h;@inbounds tile[ty*Int32(17)+tx+1]=input[y*w+x+17];end;JACC.sync_workgroup();x=by*Int32(16)+tx;y=bx*Int32(16)+ty
 if x<h&&y<w;@inbounds out[y*h+x+17]=tile[tx*Int32(17)+ty+1];end;nothing
end
const TEMPLATE=CUDA.zeros(UInt32,272)
function rec(input,out,w,h,gp)
 b=Int(cld(w,Int32(256)));spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=b,shmem_size=0,sync=false)
 JACC.parallel_for(recursive_kernel!,spec,b*256,input,out,w,h,gp)
end
function trans(input,out,w,h)
 bx=Int(cld(w,Int32(16)));by=Int(cld(h,Int32(16)));spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(16,16),blocks=(bx,by),shmem_size=1088,sync=false)
 JACC.parallel_for(transpose_kernel!,spec,(bx*16,by*16),input,out,w,h,TEMPLATE)
end
run_study(rec,trans)

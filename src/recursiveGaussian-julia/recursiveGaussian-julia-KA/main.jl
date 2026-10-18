
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function recursive_kernel!(input,out,w,h,gp)
 x=Int32(@index(Global,Linear))-Int32(1)
 recursive_body!(x,view(input,17:length(input)-16),view(out,17:length(out)-16),w,h,gp)
end
@kernel function transpose_kernel!(input,out,w,h)
 localpos=@index(Local,Cartesian);blockpos=@index(Group,Cartesian);tx=Int32(localpos[1])-Int32(1);ty=Int32(localpos[2])-Int32(1);bx=Int32(blockpos[1])-Int32(1);by=Int32(blockpos[2])-Int32(1);tile=@localmem UInt32 (272,);x=bx*Int32(16)+tx;y=by*Int32(16)+ty
 if x<w&&y<h;@inbounds tile[ty*Int32(17)+tx+1]=input[y*w+x+17];end;@synchronize;x=by*Int32(16)+tx;y=bx*Int32(16)+ty
 if x<h&&y<w;@inbounds out[y*h+x+17]=tile[tx*Int32(17)+ty+1];end
end
const BACKEND=KernelAbstractions.get_backend(CUDA.zeros(UInt8,1))
rec(input,out,w,h,gp)=recursive_kernel!(BACKEND,256)(input,out,w,h,gp;ndrange=Int(cld(w,Int32(256)))*256)
trans(input,out,w,h)=transpose_kernel!(BACKEND,(16,16))(input,out,w,h;ndrange=(Int(cld(w,Int32(16)))*16,Int(cld(h,Int32(16)))*16))
run_study(rec,trans)


include(joinpath(@__DIR__,"..","study_common.jl"))
function recursive_kernel!(input,out,w,h,gp)
 x=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 recursive_body!(x,view(input,17:length(input)-16),view(out,17:length(out)-16),w,h,gp);return
end
function transpose_kernel!(input,out,w,h)
 tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1);tile=CuStaticSharedArray(UInt32,272);x=bx*Int32(16)+tx;y=by*Int32(16)+ty
 if x<w&&y<h;@inbounds tile[ty*Int32(17)+tx+1]=input[y*w+x+17];end;sync_threads();x=by*Int32(16)+tx;y=bx*Int32(16)+ty
 if x<h&&y<w;@inbounds out[y*h+x+17]=tile[tx*Int32(17)+ty+1];end;return
end
rec(input,out,w,h,gp)=(@cuda threads=256 blocks=cld(w,Int32(256)) recursive_kernel!(input,out,w,h,gp))
trans(input,out,w,h)=(@cuda threads=(16,16) blocks=(cld(w,Int32(16)),cld(h,Int32(16))) transpose_kernel!(input,out,w,h))
run_study(rec,trans)

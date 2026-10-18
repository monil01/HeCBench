using CUDA
include("../study_common.jl")
function cost_kernel(p,L,U,R,W,H,cw)
 t=Int32(threadIdx().x-1)+Int32(32)*Int32(threadIdx().y-1);bx=Int32(blockIdx().x-1);by=Int32(blockIdx().y-1);cache=CuStaticSharedArray(Int32,768);cost_load!(cache,p,t,bx,by,W,H,cw,false);sync_threads();cost_store!(cache,L,U,R,t,bx,by,W,H,cw);nothing
end
function dp_kernel(L,U,R,M,W,H,cw)
 t=Int32(threadIdx().x-1);cache=CuDynamicSharedArray(Int32,2*cw);dp_initialize!(cache,L,U,R,M,t);sync_threads();base=Int32(0)
 for row in Int32(1):H-Int32(1);dp_row!(cache,L,U,R,M,t,W,cw,row,base);base=cw-base;sync_threads();end;nothing
end
function reduce_kernel(M,indices,cw,base)
 t=Int32(threadIdx().x-1);cache=CuStaticSharedArray(Int32,256);reduce_load!(cache,M,indices,t,cw,base);sync_threads();stride=Int32(64)
 while stride>0;reduce_step!(cache,t,stride);sync_threads();stride÷=Int32(2);end;@inbounds if t==0;indices[1]=cache[129];end;nothing
end
function backtrack_kernel(M,indices,seam,W,H,cw);backtrack!(M,indices,seam,W,H,cw);nothing;end
function remove_kernel(p,out,seam,W,H,cw)
 col=Int32((blockIdx().x-1)*32+threadIdx().x-1);row=Int32((blockIdx().y-1)*8+threadIdx().y-1);remove_lane!(p,out,seam,col,row,W,H,cw);nothing
end
function update_kernel(p,L,U,R,LS,US,RS,seam,W,H,cw)
 col=Int32((blockIdx().x-1)*32+threadIdx().x-1);row=Int32((blockIdx().y-1)*8+threadIdx().y-1);update_lane!(p,L,U,R,LS,US,RS,seam,col,row,W,H,cw);nothing
end
function approx_setup_kernel(p,index,offset,M,W,H,cw)
 t=Int32(threadIdx().x-1)+Int32(32)*Int32(threadIdx().y-1);bx=Int32(blockIdx().x-1);by=Int32(blockIdx().y-1);cache=CuStaticSharedArray(Int32,1152);cost_load!(cache,p,t,bx,by,W,H,cw,true);sync_threads();approx_costs!(cache,t,bx,by,W,H,cw);sync_threads();approx_select!(cache,index,offset,M,t,bx,by,W,H,cw);nothing
end
function approx_sum_kernel(offset,M,W,H,cw,step)
 col=Int32((blockIdx().x-1)*128+threadIdx().x-1);by=Int32(blockIdx().y-1);approx_sum_lane!(offset,M,col,by,W,H,cw,step);nothing
end
function approx_backtrack_kernel(index,indices,seam,W,H);approx_backtrack!(index,indices,seam,W,H);nothing;end
function factory(W,H,S)
 cost=(p,L,U,R,cw)->(@cuda threads=(32,8) blocks=(cld(Int(cw),30),cld(Int(H),7)) cost_kernel(p,L,U,R,W,H,cw))
 dp=(L,U,R,M,cw)->(@cuda threads=Int(cw) blocks=1 shmem=8*Int(cw) dp_kernel(L,U,R,M,W,H,cw))
 red=(M,indices,cw,base)->(@cuda threads=128 blocks=1 reduce_kernel(M,indices,cw,base))
 back=(M,indices,seam,cw)->(@cuda threads=1 blocks=1 backtrack_kernel(M,indices,seam,W,H,cw))
 remove=(p,out,seam,cw)->(@cuda threads=(32,8) blocks=(cld(Int(cw),32),cld(Int(H),8)) remove_kernel(p,out,seam,W,H,cw))
 update=(p,L,U,R,LS,US,RS,seam,cw)->(@cuda threads=(32,8) blocks=(cld(Int(cw),32),cld(Int(H),8)) update_kernel(p,L,U,R,LS,US,RS,seam,W,H,cw))
 approxsetup=(p,index,offset,M,cw)->(@cuda threads=(32,8) blocks=(cld(Int(cw),28),cld(Int(H)-1,7)) approx_setup_kernel(p,index,offset,M,W,H,cw))
 approxsum=(offset,M,cw,step,ny)->(@cuda threads=128 blocks=(cld(Int(cw),128),Int(ny)) approx_sum_kernel(offset,M,W,H,cw,step))
 approxback=(index,indices,seam)->(@cuda threads=1 blocks=1 approx_backtrack_kernel(index,indices,seam,W,H))
 (;cost,dp,red,back,remove,update,approxsetup,approxsum,approxback)
end
main()=seam_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end

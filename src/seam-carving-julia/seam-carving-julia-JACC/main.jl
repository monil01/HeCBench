using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function cost_kernel(i,p,L,U,R,W,H,cw,nx,template)
 t=Int32((i-1)%256);g=Int32((i-1)÷256);bx=g%nx;by=g÷nx;cache=JACC.shared(template);cost_load!(cache,p,t,bx,by,W,H,cw,false);JACC.sync_workgroup();cost_store!(cache,L,U,R,t,bx,by,W,H,cw)
end
function dp_kernel(i,L,U,R,M,W,H,cw,template)
 t=Int32(i-1);cache=JACC.shared(template);dp_initialize!(cache,L,U,R,M,t);JACC.sync_workgroup();base=Int32(0)
 for row in Int32(1):H-Int32(1);dp_row!(cache,L,U,R,M,t,W,cw,row,base);base=cw-base;JACC.sync_workgroup();end
end
function reduce_kernel(i,M,indices,cw,base,template)
 t=Int32(i-1);cache=JACC.shared(template);reduce_load!(cache,M,indices,t,cw,base);JACC.sync_workgroup();stride=Int32(64)
 while stride>0;reduce_step!(cache,t,stride);JACC.sync_workgroup();stride÷=Int32(2);end;@inbounds if t==0;indices[1]=cache[129];end
end
function backtrack_kernel(i,M,indices,seam,W,H,cw);backtrack!(M,indices,seam,W,H,cw);end
function remove_kernel(i,p,out,seam,W,H,cw,nx)
 t=Int32((i-1)%256);g=Int32((i-1)÷256);col=(g%nx)*Int32(32)+t%Int32(32);row=(g÷nx)*Int32(8)+t÷Int32(32);remove_lane!(p,out,seam,col,row,W,H,cw)
end
function update_kernel(i,p,L,U,R,LS,US,RS,seam,W,H,cw,nx)
 t=Int32((i-1)%256);g=Int32((i-1)÷256);col=(g%nx)*Int32(32)+t%Int32(32);row=(g÷nx)*Int32(8)+t÷Int32(32);update_lane!(p,L,U,R,LS,US,RS,seam,col,row,W,H,cw)
end
function approx_setup_kernel(i,p,index,offset,M,W,H,cw,nx,template)
 t=Int32((i-1)%256);g=Int32((i-1)÷256);bx=g%nx;by=g÷nx;cache=JACC.shared(template);cost_load!(cache,p,t,bx,by,W,H,cw,true);JACC.sync_workgroup();approx_costs!(cache,t,bx,by,W,H,cw);JACC.sync_workgroup();approx_select!(cache,index,offset,M,t,bx,by,W,H,cw)
end
function approx_sum_kernel(i,offset,M,W,H,cw,step)
 t=Int32((i-1)%128);by=Int32((i-1)÷128);approx_sum_lane!(offset,M,t,by,W,H,cw,step)
end
function approx_backtrack_kernel(i,index,indices,seam,W,H);approx_backtrack!(index,indices,seam,W,H);end
function factory(W,H,S)
 widths=collect(Int(W):-1:Int(W-S+1));slot(cw)=Int(W-cw)+1
 spec(t,b,s)=JACC.launch_spec(;threads=t,blocks=b,shmem_size=s,sync=false)
 ct=CUDA.zeros(Int32,768);rt=CUDA.zeros(Int32,256);at=CUDA.zeros(Int32,1152);dt=[CUDA.zeros(Int32,2cw) for cw in widths]
 cs=[spec(256,cld(cw,30)*cld(Int(H),7),3072) for cw in widths]
 ds=[spec(cw,1,8cw) for cw in widths]
 rs=[spec(256,cld(cw,32)*cld(Int(H),8),0) for cw in widths]
 us=[spec(256,cld(cw,32)*cld(Int(H),8),0) for cw in widths]
 aps=[spec(256,cld(cw,28)*cld(Int(H)-1,7),4608) for cw in widths]
 ss=[spec(128,cld(Int(H)-1,2step),0) for step in (1,2,4,8,16,32,64)]
 redspec=spec(128,1,1024);backspec=spec(1,1,0);abspec=spec(1,1,0)
 cost=(p,L,U,R,cw)->begin nx=cld(cw,Int32(30));JACC.parallel_for(cs[slot(cw)],Int(nx)*cld(Int(H),7)*256,cost_kernel,p,L,U,R,W,H,cw,nx,ct);end
 dp=(L,U,R,M,cw)->JACC.parallel_for(ds[slot(cw)],Int(cw),dp_kernel,L,U,R,M,W,H,cw,dt[slot(cw)])
 red=(M,indices,cw,base)->JACC.parallel_for(redspec,128,reduce_kernel,M,indices,cw,base,rt)
 back=(M,indices,seam,cw)->JACC.parallel_for(backspec,1,backtrack_kernel,M,indices,seam,W,H,cw)
 remove=(p,out,seam,cw)->begin nx=cld(cw,Int32(32));JACC.parallel_for(rs[slot(cw)],Int(nx)*cld(Int(H),8)*256,remove_kernel,p,out,seam,W,H,cw,nx);end
 update=(p,L,U,R,LS,US,RS,seam,cw)->begin nx=cld(cw,Int32(32));JACC.parallel_for(us[slot(cw)],Int(nx)*cld(Int(H),8)*256,update_kernel,p,L,U,R,LS,US,RS,seam,W,H,cw,nx);end
 approxsetup=(p,index,offset,M,cw)->begin nx=cld(cw,Int32(28));JACC.parallel_for(aps[slot(cw)],Int(nx)*cld(Int(H)-1,7)*256,approx_setup_kernel,p,index,offset,M,W,H,cw,nx,at);end
 approxsum=(offset,M,cw,step,ny)->JACC.parallel_for(ss[trailing_zeros(UInt32(step))+1],Int(ny)*128,approx_sum_kernel,offset,M,W,H,cw,step)
 approxback=(index,indices,seam)->JACC.parallel_for(abspec,1,approx_backtrack_kernel,index,indices,seam,W,H)
 (;cost,dp,red,back,remove,update,approxsetup,approxsum,approxback)
end
main()=seam_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end

using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function cost_kernel(p,L,U,R,W,H,cw,nx)
 t=Int32(@index(Local,Linear))-Int32(1);g=Int32(@index(Group,Linear))-Int32(1);bx=g%nx;by=g÷nx;cache=@localmem Int32 (768,);cost_load!(cache,p,t,bx,by,W,H,cw,false);@synchronize;cost_store!(cache,L,U,R,t,bx,by,W,H,cw)
end
@kernel function dp_kernel(L,U,R,M,W,H,cw)
 t=Int32(@index(Local,Linear))-Int32(1);cache=@localmem Int32 (256,);dp_initialize!(cache,L,U,R,M,t);@synchronize;base=Int32(0)
 for row in Int32(1):H-Int32(1);dp_row!(cache,L,U,R,M,t,W,cw,row,base);base=cw-base;@synchronize;end
end
@kernel function reduce_kernel(M,indices,cw,base)
 t=Int32(@index(Local,Linear))-Int32(1);cache=@localmem Int32 (256,);reduce_load!(cache,M,indices,t,cw,base);@synchronize;stride=Int32(64)
 while stride>0;reduce_step!(cache,t,stride);@synchronize;stride÷=Int32(2);end;@inbounds if t==0;indices[1]=cache[129];end
end
@kernel function backtrack_kernel(M,indices,seam,W,H,cw);backtrack!(M,indices,seam,W,H,cw);end
@kernel function remove_kernel(p,out,seam,W,H,cw,nx)
 t=Int32(@index(Local,Linear))-Int32(1);g=Int32(@index(Group,Linear))-Int32(1);col=(g%nx)*Int32(32)+t%Int32(32);row=(g÷nx)*Int32(8)+t÷Int32(32);remove_lane!(p,out,seam,col,row,W,H,cw)
end
@kernel function update_kernel(p,L,U,R,LS,US,RS,seam,W,H,cw,nx)
 t=Int32(@index(Local,Linear))-Int32(1);g=Int32(@index(Group,Linear))-Int32(1);col=(g%nx)*Int32(32)+t%Int32(32);row=(g÷nx)*Int32(8)+t÷Int32(32);update_lane!(p,L,U,R,LS,US,RS,seam,col,row,W,H,cw)
end
@kernel function approx_setup_kernel(p,index,offset,M,W,H,cw,nx)
 t=Int32(@index(Local,Linear))-Int32(1);g=Int32(@index(Group,Linear))-Int32(1);bx=g%nx;by=g÷nx;cache=@localmem Int32 (1152,);cost_load!(cache,p,t,bx,by,W,H,cw,true);@synchronize;approx_costs!(cache,t,bx,by,W,H,cw);@synchronize;approx_select!(cache,index,offset,M,t,bx,by,W,H,cw)
end
@kernel function approx_sum_kernel(offset,M,W,H,cw,step)
 t=Int32(@index(Local,Linear))-Int32(1);by=Int32(@index(Group,Linear))-Int32(1);approx_sum_lane!(offset,M,t,by,W,H,cw,step)
end
@kernel function approx_backtrack_kernel(index,indices,seam,W,H);approx_backtrack!(index,indices,seam,W,H);end
function factory(W,H,S)
 backend=CUDABackend();kc=cost_kernel(backend,256);kd=dp_kernel(backend);kr=reduce_kernel(backend,128);kb=backtrack_kernel(backend,1);kremove=remove_kernel(backend,256);ku=update_kernel(backend,256);ka=approx_setup_kernel(backend,256);kas=approx_sum_kernel(backend,128);kab=approx_backtrack_kernel(backend,1)
 cost=(p,L,U,R,cw)->begin nx=cld(cw,Int32(30));kc(p,L,U,R,W,H,cw,nx;ndrange=Int(nx)*cld(Int(H),7)*256);end
 dp=(L,U,R,M,cw)->kd(L,U,R,M,W,H,cw;ndrange=Int(cw),workgroupsize=Int(cw))
 red=(M,indices,cw,base)->kr(M,indices,cw,base;ndrange=128)
 back=(M,indices,seam,cw)->kb(M,indices,seam,W,H,cw;ndrange=1)
 remove=(p,out,seam,cw)->begin nx=cld(cw,Int32(32));kremove(p,out,seam,W,H,cw,nx;ndrange=Int(nx)*cld(Int(H),8)*256);end
 update=(p,L,U,R,LS,US,RS,seam,cw)->begin nx=cld(cw,Int32(32));ku(p,L,U,R,LS,US,RS,seam,W,H,cw,nx;ndrange=Int(nx)*cld(Int(H),8)*256);end
 approxsetup=(p,index,offset,M,cw)->begin nx=cld(cw,Int32(28));ka(p,index,offset,M,W,H,cw,nx;ndrange=Int(nx)*cld(Int(H)-1,7)*256);end
 approxsum=(offset,M,cw,step,ny)->kas(offset,M,W,H,cw,step;ndrange=Int(ny)*128)
 approxback=(index,indices,seam)->kab(index,indices,seam,W,H;ndrange=1)
 (;cost,dp,red,back,remove,update,approxsetup,approxsum,approxback)
end
main()=seam_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end

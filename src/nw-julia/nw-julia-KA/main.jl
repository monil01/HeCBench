using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function nw_tile_ka!(a,r,cols,width,blk,p,upper)
 tx=@index(Local,Linear)-1;bx=@index(Group,Linear)-1
 score=@localmem Int32 (289,)
 ref=@localmem Int32 (256,)
 bix=upper ? bx : bx+width-blk;biy=upper ? blk-1-bx : width-bx-1
 index=cols*16*biy+16*bix+tx+cols+1
 north=cols*16*biy+16*bix+tx+1;west=cols*16*biy+16*bix+cols;nw=cols*16*biy+16*bix
 if tx==0;score[1]=a[nw+1];end
 for ty in 0:15;ref[ty*16+tx+1]=r[index+cols*ty+1];end
 score[(tx+1)*17+1]=a[west+cols*tx+1];score[tx+2]=a[north+1]
 @synchronize
 for m in 0:15
  if tx<=m
   row=m-tx+1;col=tx+1;k=row*17+col+1
   score[k]=max(score[k-18]+ref[(row-1)*16+col],score[k-1]-p,score[k-17]-p)
  end
  @synchronize
 end
 for m in 14:-1:0
  if tx<=m
   col=tx+16-m;row=16-tx;k=row*17+col+1
   score[k]=max(score[k-18]+ref[(row-1)*16+col],score[k-1]-p,score[k-17]-p)
  end
  @synchronize
 end
 for ty in 0:15;a[index+cols*ty+1]=score[(ty+1)*17+tx+2];end
end
function nw_launch!(a,ref,dim,p)
 width=dim÷16;kernel=nw_tile_ka!(CUDABackend(),16)
 for blk in 1:width;kernel(a,ref,Int32(dim+1),Int32(width),Int32(blk),Int32(p),true;ndrange=16blk);end
 for blk in width-1:-1:1;kernel(a,ref,Int32(dim+1),Int32(width),Int32(blk),Int32(p),false;ndrange=16blk);end
end
study_main(nw_launch!)

using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const NW_TEMPLATE=CUDA.zeros(Int32,545)
const NW_BACKEND=JACC.get_backend(:cuda)
function nw_tile_jacc!(id,a,r,cols,width,blk,p,upper,template)
 tx=(id-1)%16;bx=(id-1)÷16;shared=JACC.shared(template)
 bix=upper ? bx : bx+width-blk;biy=upper ? blk-1-bx : width-bx-1
 index=cols*16*biy+16*bix+tx+cols+1
 north=cols*16*biy+16*bix+tx+1;west=cols*16*biy+16*bix+cols;nw=cols*16*biy+16*bix
 if tx==0;shared[1]=a[nw+1];end
 for ty in 0:15;shared[289+ty*16+tx+1]=r[index+cols*ty+1];end
 shared[(tx+1)*17+1]=a[west+cols*tx+1];shared[tx+2]=a[north+1]
 JACC.sync_workgroup()
 for m in 0:15
  if tx<=m
   row=m-tx+1;col=tx+1;k=row*17+col+1
   shared[k]=max(shared[k-18]+shared[289+(row-1)*16+col],shared[k-1]-p,shared[k-17]-p)
  end
  JACC.sync_workgroup()
 end
 for m in 14:-1:0
  if tx<=m
   col=tx+16-m;row=16-tx;k=row*17+col+1
   shared[k]=max(shared[k-18]+shared[289+(row-1)*16+col],shared[k-1]-p,shared[k-17]-p)
  end
  JACC.sync_workgroup()
 end
 for ty in 0:15;a[index+cols*ty+1]=shared[(ty+1)*17+tx+2];end
 nothing
end
function nw_launch!(a,ref,dim,p)
 width=dim÷16
 for blk in 1:width
  spec=JACC.LaunchSpec{typeof(NW_BACKEND)}(;threads=16,blocks=blk,shmem_size=2180,sync=false)
  JACC.parallel_for(nw_tile_jacc!,spec,16blk,a,ref,Int32(dim+1),Int32(width),Int32(blk),Int32(p),true,NW_TEMPLATE)
 end
 for blk in width-1:-1:1
  spec=JACC.LaunchSpec{typeof(NW_BACKEND)}(;threads=16,blocks=blk,shmem_size=2180,sync=false)
  JACC.parallel_for(nw_tile_jacc!,spec,16blk,a,ref,Int32(dim+1),Int32(width),Int32(blk),Int32(p),false,NW_TEMPLATE)
 end
end
study_main(nw_launch!)

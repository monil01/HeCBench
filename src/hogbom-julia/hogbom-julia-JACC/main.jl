include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATE=JACC.zeros(UInt32,768)
peak_launch(img,n,pk)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=128,shmem_size=3072,sync=false),32768,peak_jacc!,img,n,pk,TEMPLATE)
function subtract_launch(psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
 blocks=(Int(cld(ex-sx+1,Int32(16))),Int(cld(ey-sy+1,Int32(16))));JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=blocks,shmem_size=0,sync=false),(blocks[1]*16,blocks[2]*16),subtract_jacc!,psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
end
study(peak_launch,subtract_launch,(T,n)->JACC.array(Array{T}(undef,n)))

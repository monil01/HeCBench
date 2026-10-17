include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
peak_launch(img,n,pk)=@cuda threads=256 blocks=128 peak_cuda!(img,n,pk)
function subtract_launch(psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
 @cuda threads=(16,16) blocks=(Int(cld(ex-sx+1,Int32(16))),Int(cld(ey-sy+1,Int32(16)))) subtract_cuda!(psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
end
study(peak_launch,subtract_launch,(T,n)->CuArray{T}(undef,n))

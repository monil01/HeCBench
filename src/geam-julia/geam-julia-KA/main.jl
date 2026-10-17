using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function transpose_ka!(input,output,nr::Int32,nc::Int32)
 tx=Int32(@index(Local,NTuple)[1]);ty=Int32(@index(Local,NTuple)[2]);gx=Int32(@index(Group,NTuple)[1]);gy=Int32(@index(Group,NTuple)[2])
 tile=@localmem eltype(input) (33,32)
 col=(gx-Int32(1))*Int32(32)+tx;row=(gy-Int32(1))*Int32(32)+ty
 for off in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<=nc&&row+off<=nr
   @inbounds tile[tx,ty+off]=input[(row+off-Int32(1))*nc+col]
  end
 end
 @synchronize
 outrow=(gy-Int32(1))*Int32(32)+tx;outcol=(gx-Int32(1))*Int32(32)+ty
 for off in (Int32(0),Int32(8),Int32(16),Int32(24))
  if outrow<=nr&&outcol+off<=nc
   @inbounds output[(outcol+off-Int32(1))*nr+outrow]=tile[ty+off,tx]
  end
 end
end
function factory(input,output,nr,nc)
 kernel=transpose_ka!(get_backend(input),(32,8))
 return ()->kernel(input,output,Int32(nr),Int32(nc);ndrange=(cld(nc,32)*32,cld(nr,32)*8))
end
geam_study(factory)

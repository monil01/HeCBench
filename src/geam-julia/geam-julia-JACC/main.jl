using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function transpose_jacc!(indexcol,indexrow,input,output,nr::Int32,nc::Int32,template)
 tx=Int32(mod(indexcol-1,32)+1);ty=Int32(mod(indexrow-1,8)+1);gx=Int32(div(indexcol-1,32)+1);gy=Int32(div(indexrow-1,8)+1)
 tile=JACC.shared(template)
 col=(gx-Int32(1))*Int32(32)+tx;row=(gy-Int32(1))*Int32(32)+ty
 for off in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<=nc&&row+off<=nr
   @inbounds tile[tx,ty+off]=input[(row+off-Int32(1))*nc+col]
  end
 end
 JACC.sync_workgroup()
 outrow=(gy-Int32(1))*Int32(32)+tx;outcol=(gx-Int32(1))*Int32(32)+ty
 for off in (Int32(0),Int32(8),Int32(16),Int32(24))
  if outrow<=nr&&outcol+off<=nc
   @inbounds output[(outcol+off-Int32(1))*nr+outrow]=tile[ty+off,tx]
  end
 end
end
function factory(input,output,nr,nc)
 template=CUDA.zeros(eltype(input),33,32)
 spec=JACC.launch_spec(;threads=(32,8),blocks=(cld(nc,32),cld(nr,32)),shmem_size=sizeof(eltype(input))*1056,sync=false)
 return ()->JACC.parallel_for(spec,(cld(nc,32)*32,cld(nr,32)*8),transpose_jacc!,input,output,Int32(nr),Int32(nc),template)
end
geam_study(factory)

using CUDA
import JACC
JACC.@init_backend
function convert_jacc!(y,dense,rows,cols,values,m,n)
 if y<=m
  out=@inbounds rows[y]
  for x in Int32(0):n-Int32(1)
   v=@inbounds dense[(y-1)*n+x+1]
   if v!=0f0
    @inbounds begin;cols[out+1]=Int64(x);values[out+1]=v;end
    out+=Int64(1)
   end
  end
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
convert_launch(dd,dr,dc,dv,m,n)=JACC.parallel_for(JACC.launch_spec(threads=128,blocks=1,shmem_size=0,sync=false),128,convert_jacc!,dd,dr,dc,dv,m,n)
study(convert_launch,(T,n)->JACC.zeros(T,n))

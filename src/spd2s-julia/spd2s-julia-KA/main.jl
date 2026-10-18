using CUDA,KernelAbstractions
@kernel function convert_ka!(dense,rows,cols,values,m,n)
 y=@index(Global,Linear)
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
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const CONVERT=convert_ka!(BACKEND,128)
convert_launch(dd,dr,dc,dv,m,n)=CONVERT(dd,dr,dc,dv,m,n;ndrange=Int(cld(m,Int32(128)))*128)
study(convert_launch,(T,n)->KernelAbstractions.allocate(BACKEND,T,n))

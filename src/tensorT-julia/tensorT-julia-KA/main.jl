using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(si,so,ri,ro,st_i,st_l,st_g,x,y,template)
 t=@index(Local,Linear);group=@index(Group,Linear);tile=@localmem Float64 (5900,)
 for block in group-1:60:59
  it=Int32(block);off1=Int32(0)
  @inbounds for j in 1:3
   im=trunc(Int32,Float32(it)*ri[j]);off1+=st_i[j]*(it-im*si[j]);it=im
  end
  for i in t-1:256:384
   @inbounds tile[i+1]=x[i+block*385+1]
  end
  @synchronize
  for i in t-1:256:384
   it=Int32(i);off2=Int32(0);localoff=Int32(0)
   @inbounds for j in 1:3
    im=trunc(Int32,Float32(it)*ro[j]);tmp=it-im*so[j];off2+=st_g[j]*tmp;localoff+=st_l[j]*tmp;it=im
   end
   @inbounds y[off1+off2+1]=tile[localoff+1]
  end
  @synchronize
 end
end
run!(args...)=kernel!(get_backend(args[9]),256)(args...;ndrange=256*60)
study(run!)

using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(index,si,so,ri,ro,st_i,st_l,st_g,x,y,template)
 t=mod(index-1,256)+1;group=div(index-1,256)+1;tile=JACC.shared(template)
 for block in group-1:60:59
  it=Int32(block);off1=Int32(0)
  @inbounds for j in 1:3
   im=trunc(Int32,Float32(it)*ri[j]);off1+=st_i[j]*(it-im*si[j]);it=im
  end
  for i in t-1:256:384
   @inbounds tile[i+1]=x[i+block*385+1]
  end
  JACC.sync_workgroup()
  for i in t-1:256:384
   it=Int32(i);off2=Int32(0);localoff=Int32(0)
   @inbounds for j in 1:3
    im=trunc(Int32,Float32(it)*ro[j]);tmp=it-im*so[j];off2+=st_g[j]*tmp;localoff+=st_l[j]*tmp;it=im
   end
   @inbounds y[off1+off2+1]=tile[localoff+1]
  end
  JACC.sync_workgroup()
 end
end
run!(args...)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=60,shmem_size=47200,sync=false),256*60,kernel!,args...)
study(run!)

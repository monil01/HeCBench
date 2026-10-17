using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","knn_sort_math.jl"))
@kernel function distance_kernel!(a,n,b,q,dim,dist)
 l=@index(Local,NTuple);g=@index(Group,NTuple);tx=l[1]-1;ty=l[2]-1;ar=(g[2]-1)*16;br=(g[1]-1)*16
 sa=@localmem Float32 (256,);sb=@localmem Float32 (256,);acc=0f0
 for base in 0:16:dim-1
  @inbounds sa[ty*16+tx+1]=base+ty<dim&&ar+tx<n ? a[(base+ty)*n+ar+tx+1] : 0f0
  @inbounds sb[ty*16+tx+1]=base+ty<dim&&br+tx<q ? b[(base+ty)*q+br+tx+1] : 0f0
  @synchronize
  if ar+ty<n&&br+tx<q
   for z in 0:15;@inbounds diff=sa[z*16+ty+1]-sb[z*16+tx+1];acc+=diff*diff;end
  end
  @synchronize
 end
 if ar+ty<n&&br+tx<q;@inbounds dist[(ar+ty)*q+br+tx+1]=acc;end
end
@kernel function sort_kernel!(dist,ind,q,n,k)
 i=@index(Global,Linear);study_insert!(i,dist,ind,q,n,k)
end
@kernel function sqrt_kernel!(dist,q,k)
 l=@index(Global,NTuple);x=l[1];y=l[2]
 if x<=q&&y<=k;@inbounds dist[(y-1)*q+x]=sqrt(dist[(y-1)*q+x]);end
end
function knn_call!(a,b,n,q,d,k,hd,hi)
 da=CuArray(a);db=CuArray(b);ds=CuArray{Float32}(undef,n*q);di=CuArray{Int32}(undef,k*q);backend=KernelAbstractions.get_backend(da)
 distance_kernel!(backend,(16,16))(da,n,db,q,d,ds;ndrange=(cld(q,16)*16,cld(n,16)*16))
 sort_kernel!(backend,256)(ds,di,q,n,k;ndrange=q)
 sqrt_kernel!(backend,(16,16))(ds,q,k;ndrange=(q,k))
 CUDA.synchronize();copyto!(hd,1,ds,1,k*q);copyto!(hi,di)
 for a in (da,db,ds,di);CUDA.unsafe_free!(a);end
 nothing
end
study_main(knn_call!)

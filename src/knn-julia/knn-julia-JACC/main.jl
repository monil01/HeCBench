using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","knn_sort_math.jl"))
const STUDY_TEMPLATE=CUDA.zeros(Float32,512)
@inline function distance_callback!(gi,gj,a,n,b,q,dim,dist,template)
 l=((gi-1)%16+1,(gj-1)%16+1);g=((gi-1)÷16+1,(gj-1)÷16+1);tx=l[1]-1;ty=l[2]-1;ar=(g[2]-1)*16;br=(g[1]-1)*16
 shared=JACC.shared(template);acc=0f0
 for base in 0:16:dim-1
  @inbounds shared[ty*16+tx+1]=base+ty<dim&&ar+tx<n ? a[(base+ty)*n+ar+tx+1] : 0f0
  @inbounds shared[256+ty*16+tx+1]=base+ty<dim&&br+tx<q ? b[(base+ty)*q+br+tx+1] : 0f0
  JACC.sync_workgroup()
  if ar+ty<n&&br+tx<q
   for z in 0:15;@inbounds diff=shared[z*16+ty+1]-shared[256+z*16+tx+1];acc+=diff*diff;end
  end
  JACC.sync_workgroup()
 end
 if ar+ty<n&&br+tx<q;@inbounds dist[(ar+ty)*q+br+tx+1]=acc;end
end
@inline sort_callback!(i,dist,ind,q,n,k)=study_insert!(i,dist,ind,q,n,k)
@inline function sqrt_callback!(x,y,dist,q,k)
 @inbounds dist[(y-1)*q+x]=sqrt(dist[(y-1)*q+x])
end
function knn_call!(a,b,n,q,d,k,hd,hi)
 da=CuArray(a);db=CuArray(b);ds=CuArray{Float32}(undef,n*q);di=CuArray{Int32}(undef,k*q);backend=JACC.get_backend(:cuda)
 spec1=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(cld(q,16),cld(n,16)),shmem_size=2048,sync=false)
 spec2=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(q,256),shmem_size=0,sync=false)
 spec3=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(cld(q,16),cld(k,16)),shmem_size=0,sync=false)
 JACC.parallel_for(distance_callback!,spec1,(cld(q,16)*16,cld(n,16)*16),da,n,db,q,d,ds,STUDY_TEMPLATE)
 JACC.parallel_for(sort_callback!,spec2,q,ds,di,q,n,k)
 JACC.parallel_for(sqrt_callback!,spec3,(q,k),ds,q,k)
 CUDA.synchronize();copyto!(hd,1,ds,1,k*q);copyto!(hi,di)
 for a in (da,db,ds,di);CUDA.unsafe_free!(a);end
 nothing
end
study_main(knn_call!)

using CUDA
include(joinpath(@__DIR__,"..","knn_sort_math.jl"))
function distance_kernel!(a,n,b,q,dim,dist)
 tx=Int(threadIdx().x)-1;ty=Int(threadIdx().y)-1;ar=(Int(blockIdx().y)-1)*16;br=(Int(blockIdx().x)-1)*16
 sa=CuStaticSharedArray(Float32,256);sb=CuStaticSharedArray(Float32,256);acc=0f0
 for base in 0:16:dim-1
  @inbounds sa[ty*16+tx+1]=base+ty<dim&&ar+tx<n ? a[(base+ty)*n+ar+tx+1] : 0f0
  @inbounds sb[ty*16+tx+1]=base+ty<dim&&br+tx<q ? b[(base+ty)*q+br+tx+1] : 0f0
  sync_threads()
  if ar+ty<n&&br+tx<q
   for z in 0:15;@inbounds diff=sa[z*16+ty+1]-sb[z*16+tx+1];acc+=diff*diff;end
  end
  sync_threads()
 end
 if ar+ty<n&&br+tx<q;@inbounds dist[(ar+ty)*q+br+tx+1]=acc;end
 return
end
function sort_kernel!(dist,ind,q,n,k)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x;study_insert!(i,dist,ind,q,n,k);return
end
function sqrt_kernel!(dist,q,k)
 x=(blockIdx().x-1)*blockDim().x+threadIdx().x;y=(blockIdx().y-1)*blockDim().y+threadIdx().y
 if x<=q&&y<=k;@inbounds dist[(y-1)*q+x]=sqrt(dist[(y-1)*q+x]);end
 return
end

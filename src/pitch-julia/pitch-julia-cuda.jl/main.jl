using CUDA
function sigmoid2d!(::Val{P},a,w,h,stride) where P
 x=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);y=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1)
 if x<w&&y<h
  index=P ? Int64(y)*stride+x+1 : y*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
 return
end
function sigmoid3d!(::Val{P},a,w,h,d,stride) where P
 x=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);y=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1);z=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z-Int32(1)
 if x<w&&y<h&&z<d
  index=P ? (Int64(z)*h+y)*stride+x+1 : (z*h+y)*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
 return
end

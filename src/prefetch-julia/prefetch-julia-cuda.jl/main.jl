using CUDA
function add_cuda!(n::Int32,x,y)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);stride=Int32(blockDim().x*gridDim().x)
 while i<n
  @inbounds y[i+1]+=x[i+1]
  i+=stride
 end
 return
end

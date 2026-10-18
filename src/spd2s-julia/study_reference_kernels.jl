using CUDA
function convert_cuda!(dense,rows,cols,values,m,n)
 y=Int32(threadIdx().x-1)+Int32(blockIdx().x-1)*Int32(128)
 if y<m
  out=@inbounds rows[y+1]
  for x in Int32(0):n-Int32(1)
   v=@inbounds dense[y*n+x+1]
   if v!=0f0
    @inbounds begin;cols[out+1]=Int64(x);values[out+1]=v;end
    out+=Int64(1)
   end
  end
 end
 return
end

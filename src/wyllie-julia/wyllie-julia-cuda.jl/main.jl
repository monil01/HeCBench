using CUDA
function wyllie_study!(list,n)
 index=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if index<=n
  @inbounds while true
   node=list[index];(node>>32)==-1&&break
   next=list[(node>>32)+1];(next>>32)==-1&&break
   temp=(node&Int64(0xffffffff))+(next&Int64(0xffffffff))+((next>>32)<<32)
   sync_threads()
   list[index]=temp
  end
 end
 return
end

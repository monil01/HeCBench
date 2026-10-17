using CUDA
function encode_study!(dst,src,n,fmt,sat)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n
  @inbounds x=src[i]
  @inbounds dst[i]=sat ? fp8_sat(x,fmt) : fp8_ref(x,fmt)
 end
 return
end

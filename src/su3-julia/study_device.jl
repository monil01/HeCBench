# Native k_mat_nn, original 320-byte site AoS. Original full notice: LICENSE.source.
# Uint32 storage preserves all floating payload, coordinate, parity, padding and guard bits.
@inline function su3_body!(a,b,c,n,id)
 site=id÷Int32(36)
 if site<n
  direction=(id%Int32(36))÷Int32(9)
  row=(id%Int32(9))÷Int32(3)
  col=id%Int32(3)
  rr=0f0;ii=0f0
  @inbounds for k in Int32(0):Int32(2)
   ai=site*Int32(80)+direction*Int32(18)+row*Int32(6)+k*Int32(2)+Int32(1)
   bi=direction*Int32(18)+k*Int32(6)+col*Int32(2)+Int32(1)
   ar=reinterpret(Float32,a[ai]);am=reinterpret(Float32,a[ai+Int32(1)])
   br=reinterpret(Float32,b[bi]);bm=reinterpret(Float32,b[bi+Int32(1)])
   rr+=ar*br-am*bm
   ii+=ar*bm+am*br
  end
  ci=site*Int32(80)+direction*Int32(18)+row*Int32(6)+col*Int32(2)+Int32(1)
  @inbounds c[ci]=reinterpret(UInt32,rr)
  @inbounds c[ci+Int32(1)]=reinterpret(UInt32,ii)
 end
 nothing
end

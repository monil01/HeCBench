# Direct transcription of original shared-memory tiles; zero-based arithmetic.
@inline function st_bt(r,v)
 a,b,c,d,e,f=v
 r==0 ? 4f0*a-5f0*c+e : r==1 ? -4f0*b-4f0*c+d+e : r==2 ? 4f0*b-4f0*c-d+e : r==3 ? -2f0*b-c+2f0*d+e : r==4 ? 2f0*b-c-2f0*d+e : 4f0*b-5f0*d+f
end
@inline function st_at(r,v)
 a,b,c,d,e,f=v
 r==0 ? a+b+c+d+e : r==1 ? b-c+2f0*d-2f0*e : r==2 ? b+c+4f0*d+4f0*e : b-c+8f0*d-8f0*e+f
end
@inline function st_bt_load!(s,a,x,y,z,tx,ty,part,c)
 @inbounds for r in Int32(0):Int32(5)
  s[r*768+ty*128+tx+1]=a[((x*4+r)*16+y*4+ty)*c+part*128+tx+1]
 end
end
@inline st_bt_first(s,tx,ty)=ntuple(j->st_bt(ty,ntuple(r->@inbounds(s[(r-1)*768+(j-1)*128+tx+1]),Val(6))),Val(6))
@inline function st_bt_store!(s,v,tx,ty)
 @inbounds for j in 0:5;s[ty*768+j*128+tx+1]=v[j+1];end
end
@inline st_bt_second(s,tx,ty)=ntuple(i->st_bt(ty,ntuple(j->@inbounds(s[(i-1)*768+(j-1)*128+tx+1]),Val(6))),Val(6))
@inline function st_bt_output!(t,v,x,y,part,tx,ty,c)
 @inbounds for i in 0:5;t[(i*6+ty)*16*c+(x*4+y)*c+part*128+tx+1]=v[i+1];end
end
@inline function st_at_load!(s,ip,x,y,z,tx,ty,c)
 @inbounds s[tx*6+ty+1]=ip[(tx*6+ty)*16*c+(x*4+y)*c+z+1]
end
@inline st_at_first(s,tx,ty)=tx<4 ? st_at(tx,ntuple(i->@inbounds(s[(i-1)*6+ty+1]),Val(6))) : 0f0
@inline function st_at_output!(o,s,bias,scale,x,y,z,tx,ty,c)
 if tx<4 && ty<4 && x*4+tx<14 && y*4+ty<14
  @inbounds v=scale[z+1]*st_at(ty,ntuple(j->s[tx*6+j],Val(6)))+bias[z+1]
  @inbounds o[((x*4+1+tx)*16+y*4+1+ty)*c+z+1]=max(v,0f0)
 end
end
@inline function st_outer_load!(s,a,plane,part,tx,ty,c)
 ind=ty*c+tx;off=plane*16*c+part*8*c
 @inbounds s[ind+1]=a[off+ind+1]
 if c==256;@inbounds s[ind+1025]=a[off+ind+1025];end
end
@inline function st_outer_weight!(s,w,plane,k,tx,ty,c)
 ind=ty*c+tx;nt=c==128 ? 1024 : 1024;cnt=c==128 ? 4 : 8;off=plane*c*c+k*32*c
 @inbounds for j in 0:cnt-1;s[8*c+ind+j*nt+1]=w[off+ind+j*nt+1];end
end
@inline function st_outer_chunk(s,k,tx,ty,c,second)
 acc=0f0;off=ty*c+k*32+(second ? 1024 : 0)
 @inbounds for j in Int32(0):Int32(31);acc+=s[off+j+1]*s[8*c+j*c+tx+1];end
 acc
end
@inline function st_outer_output!(o,v,v2,plane,part,tx,ty,c)
 off=plane*16*c+part*8*c+ty*c+tx
 @inbounds o[off+1]=v
 if c==256;@inbounds o[off+1025]=v2;end
end
@inline function st_one_load!(s,a,bias,scale,tile,part,tx,ty,ci,co,oc)
 ind=ty*oc+tx;wo=ci*4;oo=wo+64*oc;bo=oo+4*oc
 # Kernel1024->256 uses only16-row weights and a smaller output/bias offset.
 chunk=ci==1024 ? 16 : ci==256 ? 32 : 64;oo=wo+chunk*oc;bo=oo+4*oc
 @inbounds for j in 0:ci÷oc-1;s[ind+j*4*oc+1]=a[tile*4*ci+ind+j*4*oc+1];end
 @inbounds s[bo+tx+1]=bias[part*oc+tx+1]
 @inbounds s[bo+oc+tx+1]=scale[part*oc+tx+1]
 @inbounds s[oo+ind+1]=0f0
end
@inline function st_one_weight!(s,w,k,part,tx,ty,ci,co,oc,chunk)
 @inbounds for j in 0:chunk÷4-1
  s[ci*4+(j*4+ty)*oc+tx+1]=w[(k+j*4+ty)*co+part*oc+tx+1]
 end
end
@inline function st_one_acc!(s,k,tx,ty,ci,oc,chunk)
 oo=ci*4+chunk*oc;ind=ty*oc+tx
 @inbounds v=s[oo+ind+1]
 @inbounds for j in Int32(0):chunk-1;v+=s[ty*ci+k+j+1]*s[ci*4+j*oc+tx+1];end
 @inbounds s[oo+ind+1]=v
end
@inline function st_one_output!(o,s,tile,part,tx,ty,ci,co,oc,chunk,relu)
 oo=ci*4+chunk*oc;bo=oo+4*oc
 @inbounds v=s[bo+oc+tx+1]*s[oo+ty*oc+tx+1]+s[bo+tx+1]
 @inbounds o[(tile*4+ty)*co+part*oc+tx+1]=relu ? max(v,0f0) : v
end

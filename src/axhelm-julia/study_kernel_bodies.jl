@inline private_load(q,base,offset,::Val{NC})where NC=ntuple(f->ntuple(k->q[base+(k-1)*64+(f-1)*offset+1],Val(8)),Val(NC))
@inline private_zeros(::Val{NC})where NC=ntuple(_->ntuple(_->0f0,Val(8)),Val(NC))
@inline field_zeros(::Val{NC})where NC=ntuple(_->0f0,Val(NC))
@inline put_private(a,f,k,v)=Base.setindex(a,Base.setindex(a[f],v,k),f)
@inline function helm_body!(i,j,e,offset,ggeo,D,lambda,Q,Aq,s,::Val{NC})where NC
 lane=j*Int32(8)+i+Int32(1);@inbounds s[lane]=D[lane];base=e*Int32(512)+lane-Int32(1)
 @inbounds rQ=private_load(Q,base,offset,Val(NC));rA=private_zeros(Val(NC))
 for k in 1:8
  id=e*Int32(512)+Int32(k-1)*Int32(64)+lane;gbase=e*Int32(3584)+Int32(k-1)*Int32(64)+lane
  @inbounds g00=ggeo[gbase+512];g01=ggeo[gbase+1024];g02=ggeo[gbase+1536];g11=ggeo[gbase+2048];g12=ggeo[gbase+2560];g22=ggeo[gbase+3072];gwj=ggeo[gbase];lam0=lambda[id];lam1=lambda[id+offset]
  block_barrier()
  for field in 1:NC;@inbounds s[64*field+lane]=rQ[field][k];end
  qt=field_zeros(Val(NC))
  for m in 1:8,field in 1:NC
   @inbounds qt=Base.setindex(qt,qt[field]+s[(k-1)*8+m]*rQ[field][m],field)
  end
  block_barrier();qr=field_zeros(Val(NC));qs=field_zeros(Val(NC))
  for m in 1:8,field in 1:NC
   @inbounds qr=Base.setindex(qr,qr[field]+s[i*8+m]*s[64*field+j*8+m],field);qs=Base.setindex(qs,qs[field]+s[j*8+m]*s[64*field+(m-1)*8+i+1],field)
  end
  vertical=field_zeros(Val(NC));temp=field_zeros(Val(NC))
  for field in 1:NC
   qrbase=64*(1+NC+2*(field-1));@inbounds s[qrbase+lane]=lam0*(g00*qr[field]+g01*qs[field]+g02*qt[field]);s[qrbase+64+lane]=lam0*(g01*qr[field]+g11*qs[field]+g12*qt[field]);vertical=Base.setindex(vertical,lam0*(g02*qr[field]+g12*qs[field]+g22*qt[field]),field)
   mass=gwj*lam1*rQ[field][k]
   if NC==1;temp=Base.setindex(temp,mass,field);else;rA=put_private(rA,field,k,rA[field][k]+mass);end
  end
  block_barrier()
  for m in 1:8,field in 1:NC
   qrbase=64*(1+NC+2*(field-1))
   if NC==1
    @inbounds temp=Base.setindex(temp,temp[field]+s[(m-1)*8+j+1]*s[qrbase+64+(m-1)*8+i+1],field);rA=put_private(rA,field,m,rA[field][m]+s[(k-1)*8+m]*vertical[field]);temp=Base.setindex(temp,temp[field]+s[(m-1)*8+i+1]*s[qrbase+j*8+m],field)
   else
    @inbounds temp=Base.setindex(temp,temp[field]+s[(m-1)*8+i+1]*s[qrbase+j*8+m],field);temp=Base.setindex(temp,temp[field]+s[(m-1)*8+j+1]*s[qrbase+64+(m-1)*8+i+1],field);rA=put_private(rA,field,m,rA[field][m]+s[(k-1)*8+m]*vertical[field])
   end
  end
  for field in 1:NC;rA=put_private(rA,field,k,rA[field][k]+temp[field]);end
  if NC==1;block_barrier();end
 end
 for k in 1:8,field in 1:NC;@inbounds Aq[e*512+(k-1)*64+lane+(field-1)*offset]=rA[field][k];end
 return
end

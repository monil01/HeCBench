struct XPoint
 energy::Float64
 total::Float64
 elastic::Float64
 absorption::Float64
 fission::Float64
 nu_fission::Float64
end
@inline function xlcg(s::UInt64)
 s=(UInt64(2806196910506780709)*s+UInt64(1))&UInt64(0x7fffffffffffffff)
 s,Float64(s)/9223372036854775808.
end
@inline function xforward(seed::UInt64,n::UInt64)
 a=UInt64(2806196910506780709);c=UInt64(1);aa=UInt64(1);cc=UInt64(0)
 while n>0
  if(n&UInt64(1))!=0;aa*=a;cc=cc*a+c;end
  c*=a+UInt64(1);a*=a;n>>=1
 end
 (aa*seed+cc)&UInt64(0x7fffffffffffffff)
end
@inline function xmaterial(s)
 s,roll=xlcg(s);dist=(.140,.052,.275,.134,.154,.064,.066,.055,.008,.015,.025,.013)
 for m in Int32(0):Int32(11)
  running=0.
  for j in m:Int32(-1):Int32(1);running+=dist[j+1];end
  if roll<running;return m;end
 end
 Int32(0)
end
@inline function xsample(i)
 s=xforward(UInt64(1070),UInt64(2)*UInt64(i-1));s,e=xlcg(s);e,xmaterial(s)
end
@inline function xsearch_energy(n,e,A)
 lo=Int64(0);hi=n-Int64(1)
 while hi-lo>1
  mid=lo+(hi-lo)÷Int64(2)
  if @inbounds(A[mid+1]>e);hi=mid;else;lo=mid;end
 end
 lo
end
@inline function xsearch_nuclide(e,A,lo,hi)
 while hi-lo>1
  mid=lo+(hi-lo)÷Int64(2)
  if @inbounds(A[mid+1].energy>e);hi=mid;else;lo=mid;end
 end
 lo
end
@inline xchannels(p)= (p.total,p.elastic,p.absorption,p.fission,p.nu_fission)
@inline function xmacro(e,mat,num,concs,mats,grid,union,idx,I,G,typ,bins,M)
 cursor=Int64(-1)
 if typ==0;cursor=xsearch_energy(I*G,e,union);elseif typ==2;cursor=Int64(e/(1.0/bins));end
 totalxs=(0.,0.,0.,0.,0.)
 @inbounds for j in Int32(0):num[mat+1]-Int32(1)
  nuc=Int64(mats[mat*M+j+1]);conc=concs[mat*M+j+1];low=Int64(0)
  if typ==1
   off=nuc*G;k=xsearch_nuclide(e,grid,off,off+G-1);low=k==G-1 ? k-1 : k
  elseif typ==0
   k=Int64(idx[cursor*I+nuc+1]);low=nuc*G+k-(k==G-1 ? 1 : 0)
  else
   ulo=Int64(idx[cursor*I+nuc+1]);uhi=cursor==bins-1 ? G-1 : Int64(idx[(cursor+1)*I+nuc+1])+1;elo=grid[nuc*G+ulo+1].energy;ehi=grid[nuc*G+uhi+1].energy
   lower=e<=elo ? nuc*G : e>=ehi ? nuc*G+G-1 : xsearch_nuclide(e,grid,nuc*G+ulo,nuc*G+uhi);low=rem(lower,G)==G-1 ? lower-1 : lower
  end
  l=grid[low+1];h=grid[low+2];f=(h.energy-e)/(h.energy-l.energy);lv=xchannels(l);hv=xchannels(h)
  totalxs=ntuple(k->totalxs[k]+(hv[k]-f*(hv[k]-lv[k]))*conc,Val(5))
 end
 totalxs
end
@inline function xprimary!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M)
 e,mat=xsample(i);xs=xmacro(e,mat,num,concs,mats,grid,union,idx,I,G,typ,bins,M);maximum=-1.;best=Int32(0)
 for k in 1:5
  if xs[k]>maximum;maximum=xs[k];best=Int32(k);end
 end
 @inbounds out[i]=best
 nothing
end
@inline function xobserve!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M)
 e,mat=xsample(i);xs=xmacro(e,mat,num,concs,mats,grid,union,idx,I,G,typ,bins,M)
 @inbounds begin
 out[7*(i-1)+1]=e;out[7*(i-1)+2]=Float64(mat)
 for k in 1:5;out[7*(i-1)+2+k]=xs[k];end
 end
 nothing
end

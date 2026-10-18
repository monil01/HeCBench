# Direct scalar/complex source arithmetic; no library transport/cross-section fallback.
struct RSC
 r::Float64
 i::Float64
end
struct RSPole
 ea::RSC
 rt::RSC
 ra::RSC
 rf::RSC
 l::Int16
end
struct RSWindow
 t::Float64
 a::Float64
 f::Float64
 start::Int32
 stop::Int32
end
@inline ca(a,b)=RSC(a.r+b.r,a.i+b.i)
@inline cs(a,b)=RSC(a.r-b.r,a.i-b.i)
@inline cm(a,b)=RSC(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r)
@inline function cd(a,b)
 d=b.r*b.r+b.i*b.i
 RSC((a.r*b.r+a.i*b.i)/d,(a.i*b.r-a.r*b.i)/d)
end
@inline function rs_fast_exp(x)
 x=1.0+x*0.000244140625
 for _ in 1:12;x*=x;end
 x
end
@inline function rs_cexp(z)
 e=rs_fast_exp(z.r);RSC(e*cos(z.i),e*sin(z.i))
end
@inline function rs_W(z)
 if sqrt(z.r*z.r+z.i*z.i)<6.0
  an=(.2758402,.2245740,.1594149,.09866577,.05324414,.02505215,.01027747,.003676164,.001146494,.0003117570)
  den=(9.869604,39.47842,88.82644,157.9137,246.7401,355.3058,483.6106,631.6547,799.4380,986.9604)
  ez=rs_cexp(cm(RSC(0.,12.),z));one=RSC(1.,0.)
  w=cd(cm(RSC(0.,1.),cs(one,ez)),cm(RSC(12.,0.),z));sum=RSC(0.,0.)
  for j in 1:10
   top=cs(cm(RSC(isodd(j) ? -1. : 1.,0.),ez),one)
   bot=cs(RSC(den[j],0.),cm(RSC(144.,0.),cm(z,z)))
   sum=ca(sum,cm(RSC(an[j],0.),cd(top,bot)))
  end
  ca(w,cm(RSC(0.,81.24330),cm(z,sum)))
 else
  z2=cm(z,z)
  cm(cm(z,RSC(0.,1.)),ca(cd(RSC(.512424224754768462984202823134979415014943561548661637413182,0.),cs(z2,RSC(.275255128608410950901357962647054304017026259671664935783653,0.))),cd(RSC(.051765358792987823963876628425793170829107067780337219430904,0.),cs(z2,RSC(2.724744871391589049098642037352945695982973740328335064216346,0.)))))
 end
end
@inline function rs_sig(nuc,E,l,pseudo)
 phi=pseudo[4nuc+l+1]*sqrt(E)
 if l==1;phi-= -atan(phi)
 elseif l==2;phi-=atan(3.0*phi/(3.0-phi*phi))
 elseif l==3;phi-=atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi))
 end
 phi*=2.0;RSC(cos(phi),-sin(phi))
end
@inline function rs_micro(nuc,E,doppler,nwin,pseudo,windows,poles,maxw,maxp)
 window=unsafe_trunc(Int32,E/(1.0/nwin[nuc+1]));window==nwin[nuc+1]&&(window-=Int32(1))
 w=windows[nuc*maxw+window+1]
 st=E*w.t;sa=E*w.a;sf=E*w.f
 sig0=rs_sig(nuc,E,0,pseudo);sig1=rs_sig(nuc,E,1,pseudo);sig2=rs_sig(nuc,E,2,pseudo);sig3=rs_sig(nuc,E,3,pseudo)
 for j in w.start:w.stop-Int32(1)
  pole=poles[nuc*maxp+j+1];l=pole.l
  sig=l==0 ? sig0 : l==1 ? sig1 : l==2 ? sig2 : sig3
  term=doppler==1 ? rs_W(cm(cs(RSC(E,0.),pole.ea),RSC(.5,0.))) : cd(cd(RSC(0.,1.),cs(pole.ea,RSC(sqrt(E),0.))),RSC(E,0.))
  st+=cm(pole.rt,cm(term,sig)).r;sa+=cm(pole.ra,term).r;sf+=cm(pole.rf,term).r
 end
 st,sa,sf,st-sa
end
@inline function rs_lcg(seed)
 (UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff)
end
@inline function rs_forward(seed,n)
 a=UInt64(2806196910506780709);c=UInt64(1);anew=UInt64(1);cnew=UInt64(0)
 while n>0
  if n&UInt64(1)!=0;anew*=a;cnew=cnew*a+c;end
  c*=a+UInt64(1);a*=a;n>>=1
 end
 (anew*seed+cnew)&UInt64(0x7fffffffffffffff)
end
@inline function rs_pick(seed)
 roll=Float64(seed)/9223372036854775808.0
 dist=(.140,.052,.275,.134,.154,.064,.066,.055,.008,.015,.025,.013)
 mat=0
 for i in 0:11
  running=0.
  for j in i:-1:1;running+=dist[j+1];end
  if roll<running;mat=i;break;end
 end
 mat
end
@inline function rs_lookup(i,num,concs,mats,nwin,pseudo,windows,poles,maxw,maxp,maxn,doppler)
 seed=rs_forward(UInt64(1070),UInt64(2(i-1)));seed=rs_lcg(seed);E=Float64(seed)/9223372036854775808.0;seed=rs_lcg(seed);mat=rs_pick(seed)
 t=0.;a=0.;f=0.;e=0.
 for j in 0:num[mat+1]-1
  nuc=mats[mat*maxn+j+1];micro=rs_micro(nuc,E,doppler,nwin,pseudo,windows,poles,maxw,maxp);conc=concs[mat*maxn+j+1]
  t+=micro[1]*conc;a+=micro[2]*conc;f+=micro[3]*conc;e+=micro[4]*conc
 end
 ix=Int32(1);maximum=-floatmax(Float64)
 for j in 1:4
  v=j==1 ? t : j==2 ? a : j==3 ? f : e
  if v>maximum;maximum=v;ix=Int32(j);end
 end
 ix,E,mat,seed,t,a,f,e
end
@inline function rs_store(i,num,concs,mats,nwin,pseudo,windows,poles,out,details,n,doppler,maxw,maxp,maxn,::Val{D})where D
 if i<=n
  ix,E,mat,seed,t,a,f,e=rs_lookup(i,num,concs,mats,nwin,pseudo,windows,poles,maxw,maxp,maxn,doppler)
  if D
   b=8(i-1);details[b+1]=E;details[b+2]=Float64(mat);details[b+3]=Float64(seed&UInt64(0xffffffff));details[b+4]=Float64(seed>>32)
   details[b+5]=t;details[b+6]=a;details[b+7]=f;details[b+8]=e
  else;out[i]=ix
  end
 end
 nothing
end

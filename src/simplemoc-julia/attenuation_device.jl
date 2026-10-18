# Conditional two-moment correction; original axial fitting conventions retained.
# Shared numerical body is attributed once to reference preparation, then reused.
@inline function attenuate_site!(gid,qid,fid,flux,source,sigma,state,v,F::Int32,G::Int32,S::Int32)
 if gid<S
  Q=qid[gid+1];FAI=fid[gid+1];base=Q*F*G;target=base+FAI*G
  dz=.1f0;zin=.3f0;weight=.5f0;mu=.9f0;mu2=.3f0;ds=.7f0
  g=Int32(0)
  while g<G
   y2=source[target+g+1]
   if FAI==0
    y3=source[base+G+g+1];c0=y2;c1=(y3-y2)/dz
    v[g+1]=c0+c1*zin;v[G+g+1]=c1;v[2*G+g+1]=0
   elseif FAI==F-1
    y1=source[base+(FAI-1)*G+g+1];c0=y2;c1=(y2-y1)/dz
    v[g+1]=c0+c1*zin;v[G+g+1]=c1;v[2*G+g+1]=0
   else
    y1=source[base+(FAI-1)*G+g+1];y3=source[base+(FAI+1)*G+g+1]
    c0=y2;c1=(y1-y3)/(2f0*dz);c2=(y1-2f0*y2+y3)/(2f0*dz*dz)
    v[g+1]=c0+c1*zin+c2*zin*zin;v[G+g+1]=c1+2f0*c2*zin;v[2*G+g+1]=c2
   end
   g+=1
  end
  g=Int32(0)
  while g<G
   q0=v[g+1];q1=v[G+g+1];q2=v[2*G+g+1];sig=sigma[Q*G+g+1]
   tau=sig*ds;sig2=sig*sig;ev=1f0-exp(-tau)
   reuse=(tau*(tau-2f0)+2f0*ev)/(sig*sig2)
   sf=state[g+1]
   integral=(q0*tau+(sig*sf-q0)*ev)/sig2+q1*mu*reuse*.5f0+q2*mu2*(tau*(tau*(tau-3f0)+6f0)-6f0*ev)/(3f0*sig2*sig2)
   tally=weight*integral;flux[target+g+1]+=tally
   t1=q0*ev/sig;t2=q1*mu*(tau-ev)/sig2;t3=q2*mu2*reuse;t4=sf*(1f0-ev);state[g+1]=t1+t2+t3+t4
   v[3*G+g+1]=sig;v[4*G+g+1]=tau;v[5*G+g+1]=sig2;v[6*G+g+1]=ev;v[7*G+g+1]=reuse;v[8*G+g+1]=integral;v[9*G+g+1]=tally;v[10*G+g+1]=t1;v[11*G+g+1]=t2;v[12*G+g+1]=t3;v[13*G+g+1]=t4
   g+=1
  end
 end
 nothing
end

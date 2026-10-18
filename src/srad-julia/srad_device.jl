# Original pointwise SRAD operators, including column-major reflecting indices.
# Numerical body authored once in reference preparation; framework reductions
# and dispatch are translated separately without removing their timed work.
@inline function extract_site!(i,N,I)
 if i<N;I[i+1]=exp(I[i+1]/255f0);end
 nothing
end
@inline function prepare_site!(i,N,I,s,s2)
 if i<N;x=I[i+1];s[i+1]=x;s2[i+1]=x*x;end
 nothing
end
@inline function compress_site!(i,N,I)
 if i<N;I[i+1]=log(I[i+1])*255f0;end
 nothing
end
@inline function srad_site!(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I)
 if i<N
  row=rem(i,nr);col=div(i,nr);center=I[i+1]
  north=I[iN[row+1]+nr*col+1]-center;south=I[iS[row+1]+nr*col+1]-center;west=I[row+nr*jW[col+1]+1]-center;east=I[row+nr*jE[col+1]+1]-center
  G2=(north*north+south*south+west*west+east*east)/(center*center);L=(north+south+west+east)/center
  num=.5f0*G2-.0625f0*(L*L);den=1f0+.25f0*L;qsqr=num/(den*den);den=(qsqr-q0)/(q0*(1f0+q0));coef=1f0/(1f0+den)
  if coef<0f0;coef=0f0;elseif coef>1f0;coef=1f0;end
  dN[i+1]=north;dS[i+1]=south;dW[i+1]=west;dE[i+1]=east;c[i+1]=coef
 end
 nothing
end
@inline function update_site!(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I)
 if i<N
  row=rem(i,nr);col=div(i,nr);cn=c[i+1];cs=c[iS[row+1]+nr*col+1];cw=c[i+1];ce=c[row+nr*jE[col+1]+1]
  divergence=cn*dN[i+1]+cs*dS[i+1]+cw*dW[i+1]+ce*dE[i+1]
  I[i+1]=I[i+1]+.25f0*lambda*divergence
 end
 nothing
end

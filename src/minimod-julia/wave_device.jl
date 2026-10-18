# Native 8^3 block, radius4 halo and full PML/inner update mapping.
@inline wi(i,j,k)=((i+4)*32+j+4)*32+k+5
@inline sh(i,j,k)=(i*16+j)*16+k+1
@inline function wave_zero!(ss,tx,ty,tz)
 @inbounds begin ss[sh(tz,ty,tx)]=0f0;ss[sh(tz+8,ty+8,tx+8)]=0f0;end
 nothing
end
@inline function wave_load!(ss,u,tx,ty,tz,bx,by,bz,b)
 i=b[1]+bz*8+tz;j=b[3]+by*8+ty;k=b[5]+bx*8+tx;ii=tz+4;jj=ty+4;kk=tx+4
 @inbounds begin
  ss[sh(tz+(tz÷4)*8,jj,kk)]=u[wi(i+((tz÷4)*2-1)*4,j,k)]
  ss[sh(ii,ty+(ty÷4)*8,kk)]=u[wi(i,j+((ty÷4)*2-1)*4,k)]
  ss[sh(ii,jj,tx)]=u[wi(i,j,k-4)];ss[sh(ii,jj,tx+8)]=u[wi(i,j,k+4)]
 end
 (i,j,k,ii,jj,kk)
end
@inline function wave_update!(ss,u,v,vp,phi,eta,loc,b,c0,c,hd,::Val{PML}) where PML
 i,j,k,ii,jj,kk=loc
 if i<b[2]&&j<b[4]&&k<b[6]
  @inbounds begin
   z=wi(i,j,k);center=ss[sh(ii,jj,kk)];lap=c0*center
   for r in 1:4
    lap=muladd(c[(r-1)*3+1],ss[sh(ii+r,jj,kk)]+ss[sh(ii-r,jj,kk)],lap)
    lap=muladd(c[(r-1)*3+2],ss[sh(ii,jj+r,kk)]+ss[sh(ii,jj-r,kk)],lap)
    lap=muladd(c[(r-1)*3+3],ss[sh(ii,jj,kk+r)]+ss[sh(ii,jj,kk-r)],lap)
   end
   if PML
    e=eta[z];v[z]=((2f0*e+2f0-e*e)*center+(vp[z]*(lap+phi[z])-v[z]))/(2f0*e+1f0)
    grad=(eta[wi(i+1,j,k)]-eta[wi(i-1,j,k)])*(ss[sh(ii+1,jj,kk)]-ss[sh(ii-1,jj,kk)])*hd[1]+(eta[wi(i,j+1,k)]-eta[wi(i,j-1,k)])*(ss[sh(ii,jj+1,kk)]-ss[sh(ii,jj-1,kk)])*hd[2]+(eta[wi(i,j,k+1)]-eta[wi(i,j,k-1)])*(ss[sh(ii,jj,kk+1)]-ss[sh(ii,jj,kk-1)])*hd[3]
    phi[z]=(phi[z]-grad)/(1f0+e)
   else;v[z]=2f0*center+vp[z]*lap-v[z];end
  end
 end
 nothing
end
@inline function point_source!(v,value)
 @inbounds v[wi(12,12,12)]+=value
 nothing
end

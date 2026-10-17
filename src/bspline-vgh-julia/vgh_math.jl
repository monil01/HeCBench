# Full source value/gradient/Hessian contraction, independent of framework indexing.
@inline function vgh_eval!(n,coefs,vals,grads,hess,w,walker,xs,ys,zs,ix,iy,iz,dx,dy,dz)
    wb=walker*42;base=ix*xs+iy*ys+iz*zs+n;xx=0f0;xy=0f0;xz=0f0;yy=0f0;yz=0f0;zz=0f0;gx=0f0;gy=0f0;gz=0f0;value=0f0
    @inbounds for i in 0:3,j in 0:3
        pre20=w[wb+24+i+1]*w[wb+4+j+1];pre10=w[wb+12+i+1]*w[wb+4+j+1];pre00=w[wb+i+1]*w[wb+4+j+1];pre11=w[wb+12+i+1]*w[wb+16+j+1];pre01=w[wb+i+1]*w[wb+16+j+1];pre02=w[wb+i+1]*w[wb+28+j+1]
        q=base+i*xs+j*ys;c0=coefs[q];c1=coefs[q+zs];c2=coefs[q+2zs];c3=coefs[q+3zs]
        sum0=w[wb+8+1]*c0+w[wb+8+2]*c1+w[wb+8+3]*c2+w[wb+8+4]*c3
        sum1=w[wb+20+1]*c0+w[wb+20+2]*c1+w[wb+20+3]*c2+w[wb+20+4]*c3
        sum2=w[wb+32+1]*c0+w[wb+32+2]*c1+w[wb+32+3]*c2+w[wb+32+4]*c3
        xx+=pre20*sum0;xy+=pre11*sum0;xz+=pre10*sum1;yy+=pre02*sum0;yz+=pre01*sum1;zz+=pre00*sum2;gx+=pre10*sum0;gy+=pre01*sum0;gz+=pre00*sum1;value+=pre00*sum0
    end
    gb=walker*6012+(n-1)*3;hb=walker*18036+(n-1)*9
    @inbounds begin
        vals[walker*2003+n]=value;grads[gb+1]=gx*dx;grads[gb+2]=gy*dy;grads[gb+3]=gz*dz
        hess[hb+1]=xx*dx*dx;hess[hb+2]=xy*dx*dy;hess[hb+3]=xz*dx*dz;hess[hb+4]=xy*dx*dy;hess[hb+5]=yy*dy*dy;hess[hb+6]=yz*dy*dz;hess[hb+7]=xz*dx*dz;hess[hb+8]=yz*dy*dz;hess[hb+9]=zz*dz*dz
    end
    nothing
end

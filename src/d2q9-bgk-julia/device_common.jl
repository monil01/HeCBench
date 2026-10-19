# Translation of the source D2Q9 pull-stream/BGK/bounceback body.
# All arrays include sixteen guard elements before their logical index zero.
@inline gf(a,q,i)=@inbounds a[q][i+17]
@inline function forcing_allowed(a,ob,i,w11,w21)
    @inbounds ob[i+17]==0 && gf(a,4,i)-w11>0f0 && gf(a,7,i)-w21>0f0 && gf(a,8,i)-w21>0f0
end
@inline function step_values(a,ob,ii,jj,nx,ny,da,omega)
    csinv=3f0;cs=1f0/csinv;w1=1f0/9f0;w0=4f0*w1;w2=1f0/36f0
    w11=da*w1;w21=da*w2
    yn=(jj+1)%ny;xe=(ii+1)%nx;ys=jj==0 ? jj+ny-1 : jj-1;xw=ii==0 ? ii+nx-1 : ii-1
    iw=xw+jj*nx;ie=xe+jj*nx;iws=xw+ys*nx;ies=xe+ys*nx;ien=xe+yn*nx;iwn=xw+yn*nx
    fw=jj==ny-2 && forcing_allowed(a,ob,iw,w11,w21)
    fe=jj==ny-2 && forcing_allowed(a,ob,ie,w11,w21)
    fws=ys==ny-2 && forcing_allowed(a,ob,iws,w11,w21)
    fes=ys==ny-2 && forcing_allowed(a,ob,ies,w11,w21)
    fen=yn==ny-2 && forcing_allowed(a,ob,ien,w11,w21)
    fwn=yn==ny-2 && forcing_allowed(a,ob,iwn,w11,w21)
    s0=gf(a,1,ii+jj*nx);s1=gf(a,2,iw)+(fw ? w11 : 0f0);s2=gf(a,3,ii+ys*nx)
    s3=gf(a,4,ie)-(fe ? w11 : 0f0);s4=gf(a,5,ii+yn*nx)
    s5=gf(a,6,iws)+(fws ? w21 : 0f0);s6=gf(a,7,ies)-(fes ? w21 : 0f0)
    s7=gf(a,8,ien)-(fen ? w21 : 0f0);s8=gf(a,9,iwn)+(fwn ? w21 : 0f0)
    den=s0+s1+s2+s3+s4+s5+s6+s7+s8;recip=1f0/den
    ux=(s1+s5+s8-s3-s6-s7)*recip;uy=(s2+s5+s6-s4-s8-s7)*recip
    temp2=-(ux*ux+uy*uy)/(2f0*cs);temp1=4.5f0
    d0=w0*den*(1f0+temp2)
    d1=w1*den*(1f0+ux*csinv+(ux*ux)*temp1+temp2)
    d2=w1*den*(1f0+uy*csinv+(uy*uy)*temp1+temp2)
    d3=w1*den*(1f0-ux*csinv+(ux*ux)*temp1+temp2)
    d4=w1*den*(1f0-uy*csinv+(uy*uy)*temp1+temp2)
    d5=w2*den*(1f0+(ux+uy)*csinv+((ux+uy)*(ux+uy))*temp1+temp2)
    d6=w2*den*(1f0+(-ux+uy)*csinv+((-ux+uy)*(-ux+uy))*temp1+temp2)
    d7=w2*den*(1f0+(-ux-uy)*csinv+((-ux-uy)*(-ux-uy))*temp1+temp2)
    d8=w2*den*(1f0+(ux-uy)*csinv+((ux-uy)*(ux-uy))*temp1+temp2)
    blocked=@inbounds ob[ii+jj*nx+17]!=0
    r0=blocked ? s0 : s0+omega*(d0-s0)
    r1=blocked ? s3 : s1+omega*(d1-s1);r3=blocked ? s1 : s3+omega*(d3-s3)
    r2=blocked ? s4 : s2+omega*(d2-s2);r4=blocked ? s2 : s4+omega*(d4-s4)
    r5=blocked ? s7 : s5+omega*(d5-s5);r7=blocked ? s5 : s7+omega*(d7-s7)
    r6=blocked ? s8 : s6+omega*(d6-s6);r8=blocked ? s6 : s8+omega*(d8-s8)
    rec2=1f0/(r0+r1+r2+r3+r4+r5+r6+r7+r8)
    vx=(r1+r5+r8-r3-r6-r7)*rec2;vy=(r2+r5+r6-r4-r8-r7)*rec2
    (r0,r1,r2,r3,r4,r5,r6,r7,r8),blocked ? 0f0 : hypot(vx,vy),blocked ? Int32(0) : Int32(1)
end
@inline function store_fields!(b,idx,vals)
    @inbounds for q=1:9;b[q][idx+17]=vals[q];end
    nothing
end

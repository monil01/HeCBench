
# Source NM=4,NU=4,NOCTANT=8; dimensions retain native Int32 arithmetic.
@inline ms_sf(x,y,z)=Float64(Int32(1)<<((x+Int32(3)*y+Int32(7)*z+Int32(2))&Int32(3)))
@inline ms_ef(e)=Float64(Int32(1)<<(((e*Int32(1366)+Int32(150889))%Int32(714025))&Int32(3)))
@inline ms_uf(u)=Float64(Int32(1)<<(((u*Int32(741)+Int32(60037))%Int32(312500))&Int32(3)))
@inline ms_dir(o,b)=iszero(o&b) ? Int32(1) : Int32(-1)
@inline ms_xy(a,u,e,x,y,o,d)=a+d[6]*(u+Int32(4)*(e+d[4]*(x+d[1]*(y+d[2]*o))))+Int32(1)
@inline ms_xz(a,u,e,x,z,o,d)=a+d[6]*(u+Int32(4)*(e+d[4]*(x+d[1]*(z+d[3]*o))))+Int32(1)
@inline ms_yz(a,u,e,y,z,o,d)=a+d[6]*(u+Int32(4)*(e+d[4]*(y+d[2]*(z+d[3]*o))))+Int32(1)
@inline ms_state(m,u,x,y,e,z,d)=m+d[5]*(u+Int32(4)*(x+d[1]*(y+d[2]*(e+d[4]*z))))+Int32(1)
@inline function ms_init!(::Val{S},i,j,o,d,F) where {S}
 nx,ny,nz,ne,nm,na=d;zero=Int32(0);one=Int32(1)
 limit1=S==3 ? ny : nx;limit2=S==1 ? ny : nz
 if i<limit1 && j<limit2 && o<Int32(8)
  if S==1
   x=i;y=j;z=ms_dir(o,Int32(4))==one ? -one : nz;scale=ms_sf(x,y,z)
  elseif S==2
   x=i;z=j;y=ms_dir(o,Int32(2))==one ? -one : ny;scale=ms_sf(x,y,z)
  else
   y=i;z=j;x=ms_dir(o,one)==one ? -one : nx;scale=ms_sf(x,y,z)
  end
  for e=zero:ne-one,u=zero:Int32(3),a=zero:na-one
   value=Float64(a+one)*Float64(one<<(a&Int32(7)))*scale*ms_ef(e)*ms_uf(u)*Float64(one+o)
   idx=S==1 ? ms_xy(a,u,e,i,j,o,d) : S==2 ? ms_xz(a,u,e,i,j,o,d) : ms_yz(a,u,e,i,j,o,d)
   F[idx]=value
  end
 end
 nothing
end
@inline function ms_sweep!(o,e,d,A,M,vi,vo,xy,xz,yz,local)
 nx,ny,nz,ne,nm,na=d;zero=Int32(0);one=Int32(1)
 if o<Int32(8) && e<ne
  dx=ms_dir(o,one);dy=ms_dir(o,Int32(2));dz=ms_dir(o,Int32(4))
  for wave=zero:nx+ny+nz-Int32(3),yw=zero:ny-one,xw=zero:nx-one
   x=dx==one ? xw : nx-one-xw;y=dy==one ? yw : ny-one-yw
   zw=wave-xw-yw;z=dz==one ? zw : nz-one-zw
   if zero<=z<nz
    for u=zero:Int32(3),a=zero:na-one
     value=0.0
     for m=zero:nm-one
      value=fma(A[a+na*(m+Int32(4)*o)+one],vi[ms_state(m,u,x,y,e,z,d)],value)
     end
     local[ms_xy(a,u,e,x,y,o,d)]=value
    end
    scale=ms_sf(x,y,z);sr=1.0/scale;sxr=1.0/ms_sf(x-dx,y,z);syr=1.0/ms_sf(x,y-dy,z);szr=1.0/ms_sf(x,y,z-dz);orr=1.0/Float64(o+one)
    for a=zero:na-one,u=zero:Int32(3)
     q=ms_xy(a,u,e,x,y,o,d);r=ms_xz(a,u,e,x,z,o,d);s=ms_yz(a,u,e,y,z,o,d)
     value=(local[q]*sr+(xy[q]*.5*szr+xz[r]*.25*syr+yz[s]*(.25-1.0/Float64(one<<(a&Int32(7))))*sxr)*orr)*scale
     local[q]=value;scaled=value*Float64(o+one);xy[q]=scaled;xz[r]=scaled;yz[s]=scaled
    end
    for u=zero:Int32(3),m=zero:nm-one
     value=0.0
     for a=zero:na-one
      value=fma(M[m+Int32(4)*(a+na*o)+one],local[ms_xy(a,u,e,x,y,o,d)],value)
     end
     minisweep_add!(vo,ms_state(m,u,x,y,e,z,d),value)
    end
   end
  end
 end
 nothing
end

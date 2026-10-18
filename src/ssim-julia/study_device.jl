@inline function ssim_value(x,y,z,A,B,gx,gy)
 ux=0f0;uy=0f0;uxx=0f0;uyy=0f0;uxy=0f0
 for kz=Int32(0):Int32(6)
  for ky=Int32(0):Int32(6)
   for kx=Int32(0):Int32(6)
    i=x+kx+(y+ky)*gx+(z+kz)*gx*gy+Int32(1)
    fx=@inbounds A[i];fy=@inbounds B[i]
    ux+=fx;uy+=fy;uxx+=fx*fx;uyy+=fy*fy;uxy+=fx*fy
   end
  end
 end
 w=1f0/343f0;ux*=w;uy*=w;uxx*=w;uyy*=w;uxy*=w
 cov=343f0/342f0;vx=cov*(uxx-ux*ux);vy=cov*(uyy-uy*uy);vxy=cov*(uxy-ux*uy)
 R=1f0;K1=.01f0;K2=.03f0;C1=(K1*R)*(K1*R);C2=(K2*R)*(K2*R)
 A1=2f0*ux*uy+C1;A2=2f0*vxy+C2;B1=ux*ux+uy*uy+C1;B2=vx+vy+C2;D=B1*B2
 (A1*A2)/D
end

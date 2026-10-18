struct Cmplx64
 x::Float64
 y::Float64
end
Base.zero(::Type{Cmplx64})=Cmplx64(0.,0.)
@inline function contribution(input,img,gcf,point,tx)
 inn=@inbounds input[point+1];fx=floor(Float32(inn.x));fy=floor(Float32(inn.y));subx=Int32(floor(Float32(8*(inn.x-Float64(fx)))));suby=Int32(floor(Float32(8*(inn.y-Float64(fy)))));mx=Int32(fx);my=Int32(fy);r=0.;im=0.;a=tx-Int32(16)
 for b in Int32(-16):Int32(15)
  ix=mx+a;iy=my+b;v=@inbounds img[Int32(2080)+ix+Int32(64)*iy+1]
  if ix<0||iy<0||ix>=64||iy>=64;v=Cmplx64(0.,0.);end
  c=@inbounds gcf[Int32(1024)*(Int32(8)*suby+subx)+Int32(32)*b+a+Int32(528)+1];r+=v.x*c.x-v.y*c.y;im+=v.x*c.y+c.x*v.y
 end
 return r,im
end

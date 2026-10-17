# Native qem two-stage Float32 behavior, including known WRONG objective comparison.
# Original MIT source attribution in LICENSE. NOT a validated global quartic minimizer.
@inline function source_roundaway(x)
 t=trunc(x);abs(x-t)>=0.5f0 ? t+copysign(1f0,x) : t
end
@inline function qem_qr!(i,a)
 @inbounds begin
 A,B,C,D,E,b,c,d,theta,sqrtQ,Q,R,Qint,Rint,del,x1,x2,x3,minimum,temp=a
 b[i]=0.75f0*(B[i]/A[i]);c[i]=0.50f0*(C[i]/A[i]);d[i]=0.25f0*(D[i]/A[i])
 Q[i]=c[i]/3f0-b[i]*b[i]/9f0;R[i]=b[i]*c[i]/6f0-b[i]*b[i]*b[i]/27f0-0.5f0*d[i]
 Q[i]=source_roundaway(Q[i]*1f5)/1f5;R[i]=source_roundaway(R[i]*1f5)/1f5
 Qint[i]=Q[i]*Q[i]*Q[i];Rint[i]=R[i]*R[i];del[i]=Rint[i]+Qint[i]
 end
 return
end
@inline function qem_solve!(i,a)
 @inbounds begin
 A,B,C,D,E,b,c,d,theta,sqrtQ,Q,R,Qint,Rint,del,x1,x2,x3,minimum,temp=a
 if del[i]<=1f-5
  theta[i]=acos(R[i]/sqrt(-(Q[i]*Q[i]*Q[i])));sqrtQ[i]=2f0*sqrt(-Q[i])
  x1[i]=sqrtQ[i]*cos(theta[i]/3f0)-b[i]/3f0
  x2[i]=sqrtQ[i]*cos((theta[i]+2f0*3.1415927f0)/3f0)-b[i]/3f0
  x3[i]=sqrtQ[i]*cos((theta[i]+4f0*3.1415927f0)/3f0)-b[i]/3f0
  if x1[i]<x2[i];temp[i]=x1[i];x1[i]=x2[i];x2[i]=temp[i];end
  if x2[i]<x3[i];temp[i]=x2[i];x2[i]=x3[i];x3[i]=temp[i];end
  if x1[i]<x2[i];temp[i]=x1[i];x1[i]=x2[i];x2[i]=temp[i];end
  # Intentionally preserve native defect: integrates F rather than comparing F(x1)-F(x3).
  delta=A[i]*(x1[i]*x1[i]*x1[i]*x1[i]-x3[i]*x3[i]*x3[i]*x3[i])/4f0+B[i]*(x1[i]*x1[i]*x1[i]-x3[i]*x3[i]*x3[i])/3f0+C[i]*(x1[i]*x1[i]-x3[i]*x3[i])/2f0+D[i]*(x1[i]-x3[i])
  minimum[i]=delta<=0f0 ? x1[i] : x3[i]
 else
  x1[i]=cbrt(R[i]+sqrt(del[i]))+cbrt(R[i]-sqrt(del[i]))-b[i]/3f0;x2[i]=0f0;x3[i]=0f0;minimum[i]=x1[i]
 end
 end
 return
end

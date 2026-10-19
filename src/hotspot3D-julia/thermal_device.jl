@inline function column!(a,p,b,i::Int32,j::Int32,n::Int32,z::Int32,q)
 c=i+j*n+Int32(1);xy=n*n
 w=i==Int32(0) ? c : c-Int32(1);e=i==n-Int32(1) ? c : c+Int32(1);nn=j==Int32(0) ? c : c-n;s=j==n-Int32(1) ? c : c+n
 @inbounds begin
  t2=a[c];t3=a[c+xy];t1=t2
  b[c]=q[7]*t2+q[2]*a[w]+q[1]*a[e]+q[4]*a[s]+q[3]*a[nn]+q[6]*t1+q[5]*t3+q[8]*p[c]+q[5]*80f0
  c+=xy;w+=xy;e+=xy;nn+=xy;s+=xy
  for k in Int32(1):z-Int32(2)
   t1=t2;t2=t3;t3=a[c+xy]
   b[c]=q[7]*t2+q[2]*a[w]+q[1]*a[e]+q[4]*a[s]+q[3]*a[nn]+q[6]*t1+q[5]*t3+q[8]*p[c]+q[5]*80f0
   c+=xy;w+=xy;e+=xy;nn+=xy;s+=xy
  end
  t1=t2;t2=t3
  b[c]=q[7]*t2+q[2]*a[w]+q[1]*a[e]+q[4]*a[s]+q[3]*a[nn]+q[6]*t1+q[5]*t3+q[8]*p[c]+q[5]*80f0
 end
 return nothing
end

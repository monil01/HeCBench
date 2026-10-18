@inline function utilities(A,R,n,M,row,sample,unit)
 if unit
  @inbounds return (A[0*n+row+1]+R[0*M+sample+1],A[1*n+row+1]+R[1*M+sample+1],A[2*n+row+1]+R[2*M+sample+1],A[3*n+row+1]+R[3*M+sample+1],A[4*n+row+1]+R[4*M+sample+1],A[5*n+row+1]+R[5*M+sample+1],A[6*n+row+1]+R[6*M+sample+1],A[7*n+row+1]+R[7*M+sample+1],A[8*n+row+1]+R[8*M+sample+1],A[9*n+row+1]+R[9*M+sample+1],A[10*n+row+1]+R[10*M+sample+1],A[11*n+row+1]+R[11*M+sample+1],A[12*n+row+1]+R[12*M+sample+1],A[13*n+row+1]+R[13*M+sample+1],A[14*n+row+1]+R[14*M+sample+1],A[15*n+row+1]+R[15*M+sample+1],A[16*n+row+1]+R[16*M+sample+1],A[17*n+row+1]+R[17*M+sample+1],A[18*n+row+1]+R[18*M+sample+1],A[19*n+row+1]+R[19*M+sample+1],A[20*n+row+1]+R[20*M+sample+1])
 else
  @inbounds return (A[row*21+1]+R[sample*21+1],A[row*21+2]+R[sample*21+2],A[row*21+3]+R[sample*21+3],A[row*21+4]+R[sample*21+4],A[row*21+5]+R[sample*21+5],A[row*21+6]+R[sample*21+6],A[row*21+7]+R[sample*21+7],A[row*21+8]+R[sample*21+8],A[row*21+9]+R[sample*21+9],A[row*21+10]+R[sample*21+10],A[row*21+11]+R[sample*21+11],A[row*21+12]+R[sample*21+12],A[row*21+13]+R[sample*21+13],A[row*21+14]+R[sample*21+14],A[row*21+15]+R[sample*21+15],A[row*21+16]+R[sample*21+16],A[row*21+17]+R[sample*21+17],A[row*21+18]+R[sample*21+18],A[row*21+19]+R[sample*21+19],A[row*21+20]+R[sample*21+20],A[row*21+21]+R[sample*21+21])
 end
end
@inline function winner(w)
 best=Int32(20);val=w[21]
 for k in Int32(0):Int32(19)
  if w[k+1]>val;best=k;val=w[k+1];end
 end
 return best
end
@inline function private_body!(row,A,R,P,n,M,unit)
 if row<n
  for k in Int32(0):Int32(20);index=unit ? k*n+row+1 : row*Int32(21)+k+1;@inbounds P[index]=0.;end
  for sample in Int32(0):M-Int32(1)
   w=utilities(A,R,n,M,row,sample,unit);best=winner(w);index=unit ? best*n+row+1 : row*Int32(21)+best+1;@inbounds P[index]+=1.
  end
  for k in Int32(0):Int32(20);index=unit ? k*n+row+1 : row*Int32(21)+k+1;@inbounds P[index]/=Float64(M);end
 end
 return
end
@inline function shared_body!(row,tid,s,A,R,P,n,M)
 if row<n
  for k in Int32(0):Int32(20);@inbounds s[k*Int32(96)+tid+1]=0.;end
  for sample in Int32(0):M-Int32(1)
   for k in Int32(0):Int32(20);@inbounds s[Int32(2016)+k*Int32(96)+tid+1]=A[k*n+row+1]+R[k*M+sample+1];end
   best=Int32(20);val=@inbounds s[Int32(2016)+Int32(20)*Int32(96)+tid+1]
   for k in Int32(0):Int32(19)
    v=@inbounds s[Int32(2016)+k*Int32(96)+tid+1]
    if v>val;best=k;val=v;end
   end
   @inbounds s[best*Int32(96)+tid+1]+=1.
  end
  for k in Int32(0):Int32(20);@inbounds s[k*Int32(96)+tid+1]/=Float64(M);end
  for k in Int32(0):Int32(20);@inbounds P[k*n+row+1]=s[k*Int32(96)+tid+1];end
 end
 return
end

@inline function source_bound(i,n,clip)
 i>=n ? (clip ? n-1 : i-n) : i<0 ? (clip ? zero(i) : i+n) : i
end
@inline function extrema1_site!(start,stride,n,order,clip,A,R)
 for i in start:stride:n-1
  @inbounds x=A[i+1];ok=true
  for o in Int32(1):order
   plus=source_bound(i+o,n,clip);minus=source_bound(i-o,n,clip)
   @inbounds ok &= x>A[plus+1];ok &= x>=A[minus+1]
  end
  @inbounds R[i+1]=UInt8(ok)
 end
 nothing
end
@inline function extrema2_site!(col,row,nx,ny,order,clip,axis,A,R)
 if col<nx && row<ny
  i=row*nx+col;@inbounds x=A[i+1];ok=true
  for o in Int32(1):order
   if axis==0
    plus=source_bound(row+o,ny,clip)*nx+col;minus=source_bound(row-o,ny,clip)*nx+col
   else
    plus=row*nx+source_bound(col+o,nx,clip);minus=row*nx+source_bound(col-o,nx,clip)
   end
   @inbounds ok &= x>A[plus+1];ok &= x>=A[minus+1]
  end
  @inbounds R[i+1]=UInt8(ok)
 end
 nothing
end

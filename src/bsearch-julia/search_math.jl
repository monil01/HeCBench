@inline function search_seed(n::Int64)
 bits=Int32(0)
 while (UInt64(n)>>bits)!=0;bits+=Int32(1);end
 UInt64(1)<<(bits-Int32(1))
end
@inline function search_basic(a,z,n::Int64)
 low=Int64(0);high=n
 @inbounds while high-low>1
  mid=low+(high-low)÷2
  if z<a[mid+1];high=mid;else;low=mid;end
 end
 UInt64(low)
end
@inline function search_walk(a,z,n::Int64,p::UInt64,::Val{MODE}) where {MODE}
 @inbounds idx=a[p+UInt64(1)]<=z ? p : UInt64(0)
 p>>=1
 @inbounds while p!=0
  r=idx|p
  if MODE==2
   if r<UInt64(n)&&z>=a[r+UInt64(1)];idx=r;end
  else
   w=r<UInt64(n) ? r : UInt64(n)
   if z>=a[w+UInt64(1)];idx=r;end
  end
  p>>=1
 end
 idx
end

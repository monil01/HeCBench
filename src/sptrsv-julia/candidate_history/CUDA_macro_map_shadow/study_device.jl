# MIT LICENSE.source; original custom sptrsv_mix translated with explicit warp synchronization.
# Volatile i32 loads/stores plus source device fences, not hardware atomic operations.
@generated function vload32(p::Core.LLVMPtr{Int32,A}) where {A}
 ir="define i32 @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace($A)\n %v = load volatile i32, ptr addrspace($A) %p, align 4\n ret i32 %v }"
 :(Base.llvmcall(($ir,"entry"),Int32,Tuple{UInt},UInt(p)))
end
@generated function vstore32(p::Core.LLVMPtr{Int32,A},x::Int32) where {A}
 ir="define void @entry(i64 %x, i32 %v) { %p = inttoptr i64 %x to ptr addrspace($A)\n store volatile i32 %v, ptr addrspace($A) %p, align 4\n ret void }"
 :(Base.llvmcall(($ir,"entry"),Cvoid,Tuple{UInt,Int32},UInt(p),x))
end
@inline function ready(flags,k)
 value=vload32(pointer(flags,k));CUDA.threadfence();value
end
@inline function publish(flags,k)
 CUDA.threadfence();vstore32(pointer(flags,k),Int32(1));nothing
end
@inline function solve_body!(rp,ci,v,flags,m,b,x,map,len,localid,globalid,sums)
 warp=globalid÷Int32(32)
 warp>=len-Int32(1)&&return nothing
 lane=localid&Int32(31)
 @inbounds begin
 if map[warp+Int32(2)]>map[warp+Int32(1)]+Int32(1)
  row=map[warp+Int32(1)]+lane
  row>=m&&return nothing
  j=rp[row+Int32(1)];left=0.0
  while j<rp[row+Int32(2)]
   col=ci[j+Int32(1)]
   if ready(flags,col+Int32(1))==Int32(1)
    left+=v[j+Int32(1)]*x[col+Int32(1)];j+=Int32(1);col=ci[j+Int32(1)]
   end
   if row==col
    x[row+Int32(1)]=(b[row+Int32(1)]-left)/v[rp[row+Int32(2)]];publish(flags,row+Int32(1));j+=Int32(1)
   end
  end
 else
  row=map[warp+Int32(1)];row>=m&&return nothing
  j=rp[row+Int32(1)]+lane;sum=0.0
  while j<rp[row+Int32(2)]-Int32(1)
   col=ci[j+Int32(1)]
   if ready(flags,col+Int32(1))==Int32(1)
    sum+=x[col+Int32(1)]*v[j+Int32(1)];j+=Int32(32)
   end
  end
  sums[localid+Int32(1)]=sum;CUDA.sync_warp()
  offset=Int32(16)
  while offset>0
   if lane<offset;sums[localid+Int32(1)]+=sums[localid+offset+Int32(1)];end
   CUDA.sync_warp();offset÷=Int32(2)
  end
  if lane==0
   x[row+Int32(1)]=(b[row+Int32(1)]-sums[localid+Int32(1)])/v[rp[row+Int32(2)]];publish(flags,row+Int32(1))
  end
 end
 end
 nothing
end

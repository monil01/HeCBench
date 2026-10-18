# Restored actual maxFlops macro counts and grouped reductions, ordinary Julia arithmetic.
@generated function arithmetic_site!(data,i,nIters,v1::T,v2::T,::Val{OP},::Val{L}) where {T,OP,L}
 init=Expr(:block,:(s1=@inbounds data[i]))
 if OP==2
  push!(init.args,:(s1=(s1-s1)+T(0.999f0)))
  for k in 2:L;push!(init.args,:($(Symbol("s",k))=s1-T($(Float32((k-1)*.0001)))));end
 else
  if L>=2;push!(init.args,:(s2=T(10f0)-s1));end
  for k in 3:L;v=Float32(10-((k-1)÷2));base=isodd(k) ? :s1 : :s2;push!(init.args,:($(Symbol("s",k))=T($v)-$base));end
 end
 counts=OP in (1,3) ? (240,120,60,30) : OP==2 ? (200,100,50,25) : (160,80,40,20)
 count=counts[findfirst(==(L),(1,2,4,8))];body=Expr(:block)
 for _ in 1:count, k in 1:L
  s=Symbol("s",k)
  e=OP==1 ? :(v1-$s) : OP==2 ? :(($s*$s)*v1) : OP==3 ? :(muladd(-$s,v2,v1)) : :(muladd(-v2,$s,v1)*$s)
  push!(body.args,:($s=$e))
 end
 result=L==1 ? :s1 : L==2 ? :(s1+s2) : L==4 ? :((s1+s2)+(s3+s4)) : :(((s1+s2)+(s3+s4))+((s5+s6)+(s7+s8)))
 return quote
$init
 for j in 1:nIters
$body
 end
 @inbounds data[i]=$result
 nothing
 end
end

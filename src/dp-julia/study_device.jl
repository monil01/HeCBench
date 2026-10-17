using Atomix
@inline function thread_dot(a,b,gid,stride,groups)
 T=eltype(a);s=zero(T);index=gid-1
 @inbounds while index<groups
  offset=4index
  for j in 1:4;s=muladd(a[offset+j],b[offset+j],s);end
  index+=stride
 end
 return s
end
@inline function finish_dot!(scratch,result,lid)
 step=512
 while step>0
  if lid<=step;@inbounds scratch[lid]+=scratch[lid+step];end
  # Framework-specific barriers liveinwrapper, notthisunusedhelper.
  step÷=2
 end
 return
end

struct KStore
 tag::Int32
 metric::Int32
 time::UInt64
 value::Float32
 padding::UInt32
end
struct KResult
 count::Int32
 mean::Float32
 m2::Float32
 m3::Float32
 m4::Float32
end
Base.zero(::Type{KResult})=KResult(Int32(0),0f0,0f0,0f0,0f0)
@inline function kmerge(x::KResult,y::KResult)
 x.count==0&&return y
 y.count==0&&return x
 count=x.count+y.count;n=Float32(count);n2=n*n;n3=n2*n;delta=y.mean-x.mean;d2=delta*delta;d3=d2*delta;d4=d3*delta;a=x.count;b=y.count
 mean=x.mean+delta*b/n
 m2=x.m2+y.m2;m2+=d2*a*b/n
 m3=x.m3+y.m3;m3+=d3*a*b*(a-b)/n2;m3+=3f0*delta*(a*y.m2-b*x.m2)/n
 m4=x.m4+y.m4;m4+=d4*a*b*(a*a-a*b+b*b)/n3;m4+=6f0*d2*(a*a*y.m2+b*b*x.m2)/n2;m4+=4f0*delta*(a*y.m3-b*x.m3)/n
 return KResult(count,mean,m2,m3,m4)
end
@inline kload(x,i,::Val{0})=KResult(Int32(1),(@inbounds x[i].value),0f0,0f0,0f0)
@inline kload(x,i,::Val{1})=(@inbounds x[i])

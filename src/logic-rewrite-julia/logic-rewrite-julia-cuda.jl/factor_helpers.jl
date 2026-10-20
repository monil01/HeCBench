# Original sop.cuh/vectors.cuh ABI views; layout verified by probe_factor_abi.cu.
struct LRSop <: LRPod end
struct LRVecsMem <: LRPod end
struct LRSubg <: LRPod end
@inline lr_size(::Type{LRSop})=16
@inline lr_size(::Type{LRVecsMem})=32772
@inline lr_size(::Type{LRSubg})=2056
@inline function Base.getproperty(r::LRRef{LRSop},n::Symbol)
    n===:nLits && return lr_field_scalar(r,Int32,0)
    n===:nCubes && return lr_field_scalar(r,Int32,4)
    return lr_ptr(UInt32,lr_field_scalar(r,UInt64,8))
end
@inline function Base.setproperty!(r::LRRef{LRSop},n::Symbol,v)
    n===:nLits && return lr_field_store!(r,v,Int32,0)
    n===:nCubes && return lr_field_store!(r,v,Int32,4)
    lr_field_store!(r,UInt64(UInt(getfield(v,:base))),UInt64,8);return v
end
@inline function Base.getproperty(r::LRRef{LRVecsMem},n::Symbol)
    n===:nSize && return lr_field_scalar(r,Int32,0)
    return LRPtr{UInt32}(getfield(r,:base)+4)
end
@inline Base.setproperty!(r::LRRef{LRVecsMem},n::Symbol,v)=lr_field_store!(r,v,Int32,0)
@inline function Base.getproperty(r::LRRef{LRSubg},n::Symbol)
    n===:nSize && return lr_field_scalar(r,Int32,0)
    return LRPtr{UInt64}(getfield(r,:base)+8)
end
@inline Base.setproperty!(r::LRRef{LRSubg},n::Symbol,v)=lr_field_store!(r,v,Int32,0)
@inline lr_pointer_cast(::Type{T},x::Integer) where T = lr_ptr(T,lr_cast(UInt64,x))
@inline Base.isless(a::LRPtr{T},b::LRPtr{T}) where T = UInt(getfield(a,:base))<UInt(getfield(b,:base))
@inline Base.:<(a::LRPtr{T},b::LRPtr{T}) where T = isless(a,b)
@inline Base.:<=(a::LRPtr{T},b::LRPtr{T}) where T = !isless(b,a)
@inline function lr_factor_fetch(mem,n)
    n<=0 && return lr_ptr(UInt32,UInt64(0))
    CUDA.@cuassert mem.nSize+n<=8192
    mem.nSize+=n;return mem.pArray+mem.nSize-n
end
@inline function lr_factor_shrink(mem,n)
    CUDA.@cuassert mem.nSize>=n
    mem.nSize=n;return nothing
end
@inline lr_factor_TruthWordNum(n)=n<=5 ? Int32(1) : Int32(1)<<(n-Int32(5))
@inline lr_factor_Truth6WordNum(n)=n<=6 ? Int32(1) : Int32(1)<<(n-Int32(6))
@inline lr_factor___syncthreads()=CUDA.sync_threads()
@inline lr_factor___clz(x)=Int32(leading_zeros(lr_cast(UInt32,x)))
@inline lr_factor___popc(x)=Int32(count_ones(lr_cast(UInt32,x)))
@inline function lr_factor_unbindAndNodeKeyFlag(key,a,b,c)
    c[0]=lr_cast(Int32,key&UInt64(1));b[0]=lr_cast(Int32,(key&UInt64(0xffffffff))>>1);a[0]=lr_cast(Int32,key>>32);return nothing
end
@inline function lr_factor_formAndNodeKeyFlag(a,b,c)
    CUDA.@cuassert a>=0&&b>=0&&(c==0||c==1)&&a<=b
    return (UInt64(lr_cast(UInt32,a))<<32)|UInt64((lr_cast(UInt32,b)<<1)|lr_cast(UInt32,c))
end
@inline function lr_factor_createSingleExistingVar(g,q)
    old=g.nSize;g.nSize=old+Int32(1);g.pArray[old]=lr_factor_formAndNodeKeyFlag(q,q,q&Int32(1));return nothing
end
@inline lr_factor_createConst0(g)=lr_factor_createSingleExistingVar(g,Int32(1))
@inline lr_factor_createConst1(g)=lr_factor_createSingleExistingVar(g,Int32(0))
@inline function lr_factor_addNodeAnd(g,a,b)
    CUDA.@cuassert g.nSize<256
    if a>b;a,b=b,a;end
    old=g.nSize;g.nSize=old+Int32(1);g.pArray[old]=lr_factor_formAndNodeKeyFlag(a,b,Int32(0));return (g.nSize-Int32(1))<<1
end
@inline function lr_factor_addNodeOr(g,a,b)
    CUDA.@cuassert g.nSize<256
    if a>b;a,b=b,a;end
    old=g.nSize;g.nSize=old+Int32(1);g.pArray[old]=lr_factor_formAndNodeKeyFlag(xor(a,Int32(1)),xor(b,Int32(1)),Int32(1));return ((g.nSize-Int32(1))<<1)|Int32(1)
end
@inline function lr_factor_isConst(g)
    key=g.pArray[g.nSize-Int32(1)];c=lr_cast(Int32,key&UInt64(1));b=lr_cast(Int32,(key&UInt64(0xffffffff))>>1);a=lr_cast(Int32,key>>32)
    return Int32((a>>1)==0&&a==b&&c==(a&Int32(1)))
end

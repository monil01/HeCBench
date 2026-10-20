# C object-representation adapter for genuinely authored Julia device bodies.
# No native device kernel or source PTX is loaded by this adapter.
const LRLLVMPtr = CUDA.CUDACore.LLVMPtr
abstract type LRPod end
struct LRCut <: LRPod end
struct LRTableNode <: LRPod end
struct LRLibrary <: LRPod end
struct LRArray{T,N} <: LRPod end
@inline lr_size(::Type{T}) where T = sizeof(T)
@inline lr_size(::Type{LRCut}) = 28
@inline lr_size(::Type{LRTableNode}) = 8
@inline lr_size(::Type{LRLibrary}) = 1019472
@inline lr_size(::Type{LRArray{T,N}}) where {T,N} = lr_size(T)*N
@inline lr_byte(p::Ptr) = reinterpret(Ptr{UInt8},p)
@inline lr_byte(p::LRLLVMPtr{T,A}) where {T,A} = reinterpret(LRLLVMPtr{UInt8,A},p)
@inline lr_retype(::Type{T},p::Ptr) where T = reinterpret(Ptr{T},p)
@inline lr_retype(::Type{T},p::LRLLVMPtr{U,A}) where {T,U,A} = reinterpret(LRLLVMPtr{T,A},p)
struct LRPtr{T,P}
    base::P
end
@inline LRPtr{T}(p::P) where {T,P} = LRPtr{T,typeof(lr_byte(p))}(lr_byte(p))
@inline lr_ptr(::Type{T},address::UInt64) where T = LRPtr{T}(Ptr{UInt8}(address))
@inline lr_offset(p::LRPtr{T},i::Integer) where T = LRPtr{T}(getfield(p,:base)+Int(i)*lr_size(T))
@inline Base.:+(p::LRPtr,i::Integer) = lr_offset(p,i)
@inline Base.:-(p::LRPtr,i::Integer) = lr_offset(p,-i)
@inline lr_cast(::Type{T},x::Integer) where {T<:Integer} = x % T
@inline lr_cast(::Type{Bool},x) = !iszero(x)
@inline lr_cast(::Type{T},x) where T = T(x)
@inline Base.getindex(p::LRPtr{T},i::Integer) where T = unsafe_load(lr_retype(T,getfield(lr_offset(p,i),:base)))
@inline Base.setindex!(p::LRPtr{T},v,i::Integer) where T = unsafe_store!(lr_retype(T,getfield(lr_offset(p,i),:base)),lr_cast(T,v))
struct LRRef{T,P}
    base::P
end
@inline LRRef{T}(p) where T = LRRef{T,typeof(lr_byte(p))}(lr_byte(p))
@inline Base.getindex(p::LRPtr{T},i::Integer) where {T<:LRPod} = LRRef{T}(getfield(lr_offset(p,i),:base))
@inline function lr_copy!(d,s,::Type{T}) where T
    dp=lr_retype(UInt8,getfield(d,:base));sp=lr_retype(UInt8,getfield(s,:base))
    for i in 1:lr_size(T);unsafe_store!(dp,unsafe_load(sp,i),i);end
    return d
end
@inline Base.setindex!(p::LRPtr{T},v::LRRef{T},i::Integer) where {T<:LRPod} = lr_copy!(lr_offset(p,i),v,T)
@inline function lr_field_scalar(r,::Type{T},offset) where T
    return LRPtr{T}(getfield(r,:base)+offset)[0]
end
@inline function lr_field_store!(r,v,::Type{T},offset) where T
    LRPtr{T}(getfield(r,:base)+offset)[0]=v;return v
end
@inline function Base.getproperty(r::LRRef{LRCut},name::Symbol)
    name===:sign && return lr_field_scalar(r,Int32,0)
    name===:leaves && return LRPtr{Int32}(getfield(r,:base)+4)
    name===:used && return lr_field_scalar(r,UInt8,24)!=0
    word=lr_field_scalar(r,UInt32,20)
    name===:truthtable && return word & UInt32(0xffff)
    name===:value && return (word>>16)&UInt32(0x7ff)
    return (word>>27)&UInt32(0xf) # nLeaves; callers use only source-declared fields.
end
@inline function Base.setproperty!(r::LRRef{LRCut},name::Symbol,v)
    name===:sign && return lr_field_store!(r,v,Int32,0)
    name===:used && return lr_field_store!(r,UInt8(v!=0),UInt8,24)
    word=lr_field_scalar(r,UInt32,20)
    shift=name===:truthtable ? 0 : name===:value ? 16 : 27
    mask=name===:truthtable ? UInt32(0xffff) : name===:value ? UInt32(0x7ff) : UInt32(0xf)
    out=(word & ~(mask<<shift))|((lr_cast(UInt32,v)&mask)<<shift)
    lr_field_store!(r,out,UInt32,20);return v
end
@inline function Base.getproperty(r::LRRef{LRTableNode},name::Symbol)
    lr_field_scalar(r,Int32,name===:val ? 0 : 4)
end
@inline function Base.setproperty!(r::LRRef{LRTableNode},name::Symbol,v)
    lr_field_store!(r,v,Int32,name===:val ? 0 : 4)
end
@inline function lr_countones(uWord::UInt32)
    uWord=(uWord&UInt32(0x55555555))+((uWord>>1)&UInt32(0x55555555))
    uWord=(uWord&UInt32(0x33333333))+((uWord>>2)&UInt32(0x33333333))
    uWord=(uWord&UInt32(0x0f0f0f0f))+((uWord>>4)&UInt32(0x0f0f0f0f))
    uWord=(uWord&UInt32(0x00ff00ff))+((uWord>>8)&UInt32(0x00ff00ff))
    return Int32((uWord&UInt32(0xffff))+(uWord>>16))
end
@inline function lr_free(p::Ptr{Cvoid})
    Base.llvmcall(("""
        declare void @free(i8*)
        define void @entry(i64 %address) {
            %p = inttoptr i64 %address to i8*
            call void @free(i8* %p)
            ret void
        }
        ""","entry"),Cvoid,Tuple{UInt64},UInt64(UInt(p)))
end
@inline lr_alloc(::Type{T},n::Integer=1) where T = LRPtr{T}(CUDA.malloc(Csize_t(lr_size(T)*Int(n))))
@inline lr_truth(x::Number) = x!=0
@inline lr_truth(p::LRPtr) = UInt(getfield(p,:base))!=0
@inline lr_pointer_cast(::Type{T},p::Union{LRPtr,LRRef}) where T = LRPtr{T}(getfield(p,:base))
@inline Base.getproperty(p::LRPtr{T},name::Symbol) where {T<:LRPod} = getproperty(p[0],name)
@inline Base.setproperty!(p::LRPtr{T},name::Symbol,v) where {T<:LRPod} = setproperty!(p[0],name,v)
@inline Base.getindex(r::LRRef{LRArray{T,N}},i::Integer) where {T,N} = LRPtr{T}(getfield(r,:base))[i]
@inline Base.setindex!(r::LRRef{LRArray{T,N}},v,i::Integer) where {T,N} = (LRPtr{T}(getfield(r,:base))[i]=v)
@inline lr_cast(::Type{Bool},x::Integer) = x!=0
@inline lr_field_address(p::LRPtr{T},v::Val) where {T<:LRPod} = lr_field_address(p[0],v)
@inline lr_field_address(r::LRRef{LRTableNode},::Val{:val}) = LRPtr{Int32}(getfield(r,:base))
@inline lr_field_address(r::LRRef{LRTableNode},::Val{:next}) = LRPtr{Int32}(getfield(r,:base)+4)
@inline function lr_atomic_pointer(p::LRPtr{T}) where T
    reinterpret(LRLLVMPtr{T,0},UInt(getfield(p,:base)))
end
@inline lr_atomicCAS(p::LRPtr{T},old,new) where T = CUDA.atomic_cas!(lr_atomic_pointer(p),lr_cast(T,old),lr_cast(T,new))
@inline lr_atomicAdd(p::LRPtr{T},v) where T = CUDA.atomic_add!(lr_atomic_pointer(p),lr_cast(T,v))

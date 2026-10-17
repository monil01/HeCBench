# Packed64size/stride metadata and nestedrowaccessor, sourceTensorAccessorsemantics.
# OriginaltensorAccessor-cuda redistributionterms preservedinLICENSE.
struct StudyPacked{N,A}
 data::A
 sizes::NTuple{N,Int64}
 strides::NTuple{N,Int64}
end
struct StudyRow{A}
 data::A
 offset::Int64
 n::Int64
 stride::Int64
end
@inline Base.size(a::StudyPacked,d::Integer)=a.sizes[d]
@inline Base.getindex(a::StudyPacked{2},i)=StudyRow(a.data,Int64(i-1)*a.strides[1],a.sizes[2],a.strides[2])
@inline Base.getindex(a::StudyRow,j)=(@inbounds a.data[1+a.offset+Int64(j-1)*a.stride])
@inline Base.getindex(a::StudyPacked{1},i)=(@inbounds a.data[1+Int64(i-1)*a.strides[1]])
@inline function Base.setindex!(a::StudyPacked{1},value,i)
 @inbounds a.data[1+Int64(i-1)*a.strides[1]]=value
 value
end
# Extend the existing pinnedAdapt module; noenvironment mutation/newdependency.
CUDA.CUDACore.Adapt.adapt_structure(to,a::StudyPacked)=StudyPacked(CUDA.CUDACore.Adapt.adapt(to,a.data),a.sizes,a.strides)

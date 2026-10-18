const HDT=reinterpret(Float32,UInt32(994960175))
# VecElement payload preserves native packed float4/float2 physical representation.
struct StudyV{N}
 data::NTuple{N,VecElement{Float32}}
end
const StudyV4=StudyV{4}
const StudyV2=StudyV{2}
@inline StudyV{N}(x::Float32) where N=StudyV{N}(ntuple(_->VecElement(x),Val(N)))
@inline StudyV2(x::Float32,y::Float32)=StudyV2((VecElement(x),VecElement(y)))
@inline Base.:+(a::StudyV{N},b::StudyV{N}) where N=StudyV{N}(ntuple(i->VecElement(a.data[i].value+b.data[i].value),Val(N)))
@inline Base.:-(a::StudyV{N},b::StudyV{N}) where N=StudyV{N}(ntuple(i->VecElement(a.data[i].value-b.data[i].value),Val(N)))
@inline Base.:*(a::StudyV{N},b::StudyV{N}) where N=StudyV{N}(ntuple(i->VecElement(a.data[i].value*b.data[i].value),Val(N)))
@inline Base.:*(a::Float32,b::StudyV{N}) where N=StudyV{N}(ntuple(i->VecElement(a*b.data[i].value),Val(N)))
@inline Base.:*(a::StudyV{N},b::Float32) where N=b*a

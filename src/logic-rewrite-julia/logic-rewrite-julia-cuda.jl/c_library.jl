# Source/CPUClangABI observed library field views; no device function fallback.
@inline function Base.getproperty(r::LRRef{LRLibrary},name::Symbol)
    name===:pPhases && return LRPtr{Int32}(getfield(r,:base)+0)
    name===:pPerms && return LRPtr{Int32}(getfield(r,:base)+262144)
    name===:pPerms4 && return LRPtr{LRArray{Int32,4}}(getfield(r,:base)+524288)
    name===:pMap && return LRPtr{Int32}(getfield(r,:base)+524672)
    name===:nNodes && return LRPtr{Int32}(getfield(r,:base)+786816)
    name===:nSubgr && return LRPtr{Int32}(getfield(r,:base)+787704)
    name===:pSubgr && return LRPtr{LRArray{Int32,20}}(getfield(r,:base)+788592)
    name===:fanin0 && return LRPtr{LRArray{Int32,60}}(getfield(r,:base)+806352)
    name===:fanin1 && return LRPtr{LRArray{Int32,60}}(getfield(r,:base)+859632)
    name===:isC0 && return LRPtr{LRArray{Int32,60}}(getfield(r,:base)+912912)
    name===:isC1 && return LRPtr{LRArray{Int32,60}}(getfield(r,:base)+966192)
    CUDA.@cuassert false
    return LRPtr{Int32}(getfield(r,:base))
end

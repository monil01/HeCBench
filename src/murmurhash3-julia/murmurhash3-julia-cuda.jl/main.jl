using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function murmur_kernel!(keys, offsets, lengths, out, num_keys::UInt32)
    i0 = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if i0 < num_keys
        idx = i0 + UInt32(1)
        @inbounds base = offsets[idx]
        @inbounds len = lengths[idx]
        h1, h2 = murmur_hash(keys, base, len, i0)
        @inbounds begin
            out[i0 * UInt32(2) + UInt32(1)] = h1
            out[i0 * UInt32(2) + UInt32(2)] = h2
        end
    end
    return
end

function factory(keys,offsets,lengths,out,n)
 kernel=@cuda launch=false murmur_kernel!(keys,offsets,lengths,out,UInt32(n))
 ()->kernel(keys,offsets,lengths,out,UInt32(n);threads=256,blocks=cld(n,256))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hash_study(factory)

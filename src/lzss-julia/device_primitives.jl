# Shared byte layout: UInt32 data[512], UInt8 lengths[512], UInt8 offsets[512],
# UInt32 prefix[513], UInt8 flags[64] = 5188 bytes; explicit little-endian packing.
@inline function loadword(a,o)
    @inbounds return UInt32(a[o+1]) | (UInt32(a[o+2])<<8) | (UInt32(a[o+3])<<16) | (UInt32(a[o+4])<<24)
end
@inline function storeword!(a,o,v)
    @inbounds begin
        a[o+1]=UInt8(v&0xff);a[o+2]=UInt8((v>>8)&0xff)
        a[o+3]=UInt8((v>>16)&0xff);a[o+4]=UInt8((v>>24)&0xff)
    end
    return nothing
end

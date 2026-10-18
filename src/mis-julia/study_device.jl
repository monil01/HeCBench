const IN_STATUS=UInt8(0xfe)
const OUT_STATUS=UInt8(0)
const THREADS_PER_BLOCK=256
# UInt8 adaptation of source-reviewed ECL-BH LLVM volatile wrappers.
# Volatile is NOT atomic/fence; source status protocol retained, not strengthened.
@generated function vload(p::Core.LLVMPtr{UInt8,A}) where {A}
 ir="define i8 @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace($A)\n %v = load volatile i8, ptr addrspace($A) %p, align 1\n ret i8 %v }"
 :(Base.llvmcall(($ir,"entry"),UInt8,Tuple{UInt},UInt(p)))
end
@generated function vstore(p::Core.LLVMPtr{UInt8,A},x::UInt8) where {A}
 ir="define void @entry(i64 %x, i8 %v) { %p = inttoptr i64 %x to ptr addrspace($A)\n store volatile i8 %v, ptr addrspace($A) %p, align 1\n ret void }"
 :(Base.llvmcall(($ir,"entry"),Cvoid,Tuple{UInt,UInt8},UInt(p),x))
end
@inline function hash_u32(val::UInt32)
    val = ((val >> 16) ⊻ val) * UInt32(0x045d9f3b)
    val = ((val >> 16) ⊻ val) * UInt32(0x045d9f3b)
    return (val >> 16) ⊻ val
end

@inline function init_body!(nodes::Int32, edges::Int32, nidx, nstat, from::Int32, incr::Int32)
    avg = Float32(edges) / Float32(nodes)
    scaledavg = Float32((Int32(IN_STATUS) ÷ Int32(2)) - Int32(1)) * avg

    i0 = from
        while i0 < nodes
            idx = i0 + Int32(1)
            val = IN_STATUS
            degree = @inbounds nidx[idx + Int32(1)] - nidx[idx]
        if degree > Int32(0)
            x = Float32(degree) -
                (Float32(hash_u32(UInt32(i0))) * Float32(0.00000000023283064365386962890625))
            res = unsafe_trunc(Int32, scaledavg / (avg + x))
            val = UInt8(UInt32((res + res) | Int32(1)) & UInt32(0xff))
        end
        @inbounds nstat[idx] = val
        i0 += incr
    end
    return
end

@inline function findmins_body!(nodes::Int32, nidx, nlist, nstat, from::Int32, incr::Int32)

    missing = Int32(0)
    while true
        missing = Int32(0)
        v0 = from
        while v0 < nodes
            vidx = v0 + Int32(1)
            nv = vload(pointer(nstat,vidx))
            if (nv & UInt8(0x01)) != UInt8(0)
                i = @inbounds nidx[vidx]
                stop = @inbounds nidx[vidx + Int32(1)]
                while i < stop
                    nb0 = @inbounds nlist[i + Int32(1)]
                    nbstat = vload(pointer(nstat,nb0+Int32(1)))
                    if !((nv > nbstat) || ((nv == nbstat) && (v0 > nb0)))
                        break
                    end
                    i += Int32(1)
                end
                if i < stop
                    missing = Int32(1)
                else
                    j = @inbounds nidx[vidx]
                    while j < stop
                        nb0 = @inbounds nlist[j + Int32(1)]
                        vstore(pointer(nstat,nb0+Int32(1)),OUT_STATUS)
                        j += Int32(1)
                    end
                    vstore(pointer(nstat,vidx),IN_STATUS)
                end
            end
            v0 += incr
        end
        missing == Int32(0) && break
    end
    return
end


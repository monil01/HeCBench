const N_DIRECTIONS=32
const K_2POWNEG32=2.3283064f-10
@inline ffs_u32(x::UInt32)=x==0 ? Int32(0) : Int32(trailing_zeros(x)+1)
@inline function sobol_body!(tx,bx,by,gx,v,d_output,n_vectors)
 out_off=by*n_vectors
    i0     = tx + bx * Int32(128)
    stride = gx * Int32(128)

    g = UInt32(i0) ⊻ (UInt32(i0) >> 1)

    X = UInt32(0)
    ffs_stride = ffs_u32(UInt32(stride))
    kmax = ffs_stride - Int32(1)
    @inbounds for k in Int32(0):(kmax - Int32(1))
        mask = -(g & UInt32(1))
        X ⊻= mask & v[k + Int32(1)]
        g = g >> 1
    end

    if i0 < n_vectors
        @inbounds d_output[out_off + i0 + Int32(1)] = Float32(X) * K_2POWNEG32
    end

    @inbounds v_log2stridem1 = v[ffs_stride - Int32(1)]
    v_stridemask = UInt32(stride) - UInt32(1)

    i = i0 + stride
    while i < n_vectors
        prev = UInt32(i - stride) | v_stridemask
        inv_prev = ~prev
        idx = ffs_u32(inv_prev)
        @inbounds X ⊻= v_log2stridem1 ⊻ v[idx]
        @inbounds d_output[out_off + i + Int32(1)] = Float32(X) * K_2POWNEG32
        i += stride
    end
    return
end


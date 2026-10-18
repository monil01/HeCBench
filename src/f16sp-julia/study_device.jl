# Precision semantics from native f16sp kernels.h; no packed-half2 ISA equivalence implied.
@inline function f16sp_acc(a,b,n,index,stride,::Type{T}) where T
    x=zero(T);y=zero(T);i=index
    while i<n
        @inbounds x=fma(T(a[2i+1]),T(b[2i+1]),x)
        @inbounds y=fma(T(a[2i+2]),T(b[2i+2]),y)
        i+=stride
    end
    x,y
end
@inline function f16sp_warp(value,lane)
    for off in (1,2,4,8,16)
        other=CUDA.shfl_down_sync(typemax(UInt32),value,off)
        if lane+off<32;value+=other;end
    end
    value
end

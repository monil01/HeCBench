using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function sigmoid_f32(x::Float32)
    return 1.0f0 / (1.0f0 + exp(-x))
end

function portable_gru!(idx,input,hidden,bias1,bias2,hx,hy,storage,hsz::Int32,total_elements::Int32)
    i=Int32(idx);stride=total_elements
    while i <= total_elements
        linear = i - Int32(1)
        batch_idx = linear ÷ hsz
        hidden_idx = linear - batch_idx * hsz
        offset = batch_idx * Int32(3) * hsz + hidden_idx + Int32(1)

        ir = @inbounds input[offset]
        ii = @inbounds input[offset + hsz]
        inn = @inbounds input[offset + Int32(2) * hsz]
        hr = @inbounds hidden[offset]
        hi = @inbounds hidden[offset + hsz]
        hn = @inbounds hidden[offset + Int32(2) * hsz]
        hxv = @inbounds hx[i]

        bidx = hidden_idx + Int32(1)
        b1r = @inbounds bias1[bidx]
        b1i = @inbounds bias1[bidx + hsz]
        b1n = @inbounds bias1[bidx + Int32(2) * hsz]
        b2r = @inbounds bias2[bidx]
        b2i = @inbounds bias2[bidx + hsz]
        b2n = @inbounds bias2[bidx + Int32(2) * hsz]

        rg = sigmoid_f32(Float32(ir) + Float32(hr) + Float32(b1r) + Float32(b2r))
        ig = sigmoid_f32(Float32(ii) + Float32(hi) + Float32(b1i) + Float32(b2i))
        ng = tanh(Float32(inn) + Float32(b1n) + rg * (Float32(hn) + Float32(b2n)))
        @inbounds hy[i] = Float16(ng + ig * (Float32(hxv) - ng))

        store_offset = batch_idx * Int32(5) * hsz + hidden_idx + Int32(1)
        @inbounds storage[store_offset] = Float16(rg)
        @inbounds storage[store_offset + hsz] = Float16(ig)
        @inbounds storage[store_offset + Int32(2) * hsz] = Float16(ng)
        @inbounds storage[store_offset + Int32(3) * hsz] = hxv
        @inbounds storage[store_offset + Int32(4) * hsz] = Float16(Float32(hn) + Float32(b2n))
        i += stride
    end
    nothing
end
function study_prepare(input,hidden,b1,b2,hx,hy,storage,h,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_gru!,spec,n,input,hidden,b1,b2,hx,hy,storage,Int32(h),Int32(n))
end
exit(study_main(study_prepare))

using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function lanes_for_method(method::Int32)
    return method == 1 ? Int32(32) :
           method == 2 ? Int32(64) :
           method == 3 ? Int32(128) : Int32(256)
end

function portable_spmm!(ix,iy,nrows::Int32,bncols::Int32,rowptr,colind,aval,b,c,method::Int32)
    i=Int32(ix)-Int32(1);j=Int32(iy)-Int32(1);lane=i%Int32(32);row=(i÷Int32(32))*Int32(4)+j%Int32(4);group=j÷Int32(4)
    lanes = lanes_for_method(method)
    base_col = group * lanes + lane
    stride = Int32(32)
    while base_col < min((group + Int32(1)) * lanes, bncols)
        if row < nrows
            acc = 0.0f0
            lo = @inbounds rowptr[row + 1]
            hi = @inbounds rowptr[row + 2]
            for ptr in lo:hi-Int32(1)
                col = @inbounds colind[ptr + Int32(1)]
                acc += (@inbounds aval[ptr + Int32(1)]) * (@inbounds b[col * bncols + base_col + Int32(1)])
            end
            @inbounds c[row * bncols + base_col + Int32(1)] = acc
        end
        base_col += stride
    end
    nothing
end
function study_prepare(rp,ci,av,b,c,nr,bc)
    backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};specs=[ST(;threads=(32,4),blocks=(cld(nr,4),cld(bc,32<<(method-1))),shmem_size=0,sync=false) for method in 1:4]
    method->JACC.parallel_for(portable_spmm!,specs[method],(cld(nr,4)*32,cld(bc,32<<(method-1))*4),Int32(nr),Int32(bc),rp,ci,av,b,c,Int32(method))
end
exit(study_main(study_prepare))

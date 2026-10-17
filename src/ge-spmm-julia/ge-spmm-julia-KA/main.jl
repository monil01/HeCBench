using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function lanes_for_method(method::Int32)
    return method == 1 ? Int32(32) :
           method == 2 ? Int32(64) :
           method == 3 ? Int32(128) : Int32(256)
end

@kernel function portable_spmm!(nrows::Int32,bncols::Int32,rowptr,colind,aval,b,c,method::Int32)
    gx,gy=@index(Group,NTuple);tx,ty=@index(Local,NTuple);lane=Int32(tx)-Int32(1);row=(Int32(gx)-Int32(1))*Int32(4)+Int32(ty)-Int32(1);group=Int32(gy)-Int32(1)
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
    kernel=portable_spmm!(CUDABackend(),(32,4))
    method->kernel(Int32(nr),Int32(bc),rp,ci,av,b,c,Int32(method);ndrange=(cld(nr,4)*32,cld(bc,32<<(method-1))*4))
end
exit(study_main(study_prepare))

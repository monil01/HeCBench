using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function mandel_ka!(out,p,rows::Int32,cols::Int32)
 ii,jj=@index(Global,NTuple);i=Int32(ii)-Int32(1);j=Int32(jj)-Int32(1)
 if i<rows&&j<cols;@inbounds out[Int32(16)+i*cols+j+1]=mandel_count(i,j,p[17],p[18],p[19]);end
end
function factory(out,p,rows,cols)
 k=mandel_ka!(CUDABackend(),(16,16))
 ()->k(out,p,Int32(rows),Int32(cols);ndrange=(rows,cols))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mandel_study(factory)

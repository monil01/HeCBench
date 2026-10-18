using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function mandel_cuda!(out,p,rows::Int32,cols::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1)
 if i<rows&&j<cols;@inbounds out[Int32(16)+i*cols+j+1]=mandel_count(i,j,p[17],p[18],p[19]);end
 return
end
function factory(out,p,rows,cols)
 kernel=@cuda launch=false mandel_cuda!(out,p,Int32(rows),Int32(cols))
 ()->kernel(out,p,Int32(rows),Int32(cols);threads=(16,16),blocks=(cld(rows,16),cld(cols,16)))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mandel_study(factory)

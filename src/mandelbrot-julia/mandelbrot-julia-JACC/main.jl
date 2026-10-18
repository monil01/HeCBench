using CUDA,JACC
include(joinpath(@__DIR__,"..","study_math.jl"))
function mandel_jacc!(ii,jj,out,p,rows::Int32,cols::Int32)
 i=Int32(ii)-Int32(1);j=Int32(jj)-Int32(1)
 if i<rows&&j<cols;@inbounds out[Int32(16)+i*cols+j+1]=mandel_count(i,j,p[17],p[18],p[19]);end
 return
end
function factory(out,p,rows,cols)
 spec=JACC.launch_spec(;threads=(16,16),blocks=(cld(rows,16),cld(cols,16)),shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,(rows,cols),mandel_jacc!,out,p,Int32(rows),Int32(cols))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mandel_study(factory)

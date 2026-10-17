include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,output,nr,nc)
 T=eltype(input);aliasB=reshape(input,nr,nc)
 return ()->CUDA.CUBLAS.geam!('T','N',one(T),input,zero(T),aliasB,output)
end
geam_study(factory)

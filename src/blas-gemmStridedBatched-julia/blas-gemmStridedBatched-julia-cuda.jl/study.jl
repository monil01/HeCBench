using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nmain\(\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(A,B,C,m,k,upper,batch)
    a=@view reshape(A,upper,upper,batch)[1:m,1:k,:]
    b=@view reshape(B,upper,1,batch)[1:k,1:1,:]
    c=@view reshape(C,upper,1,batch)[1:m,1:1,:]
    ()->CUDA.CUBLAS.gemm_strided_batched!('N','N',1.0,a,b,0.0,c)
end
batched_study(factory)

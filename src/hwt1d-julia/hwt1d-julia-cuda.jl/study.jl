include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"main\(\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_scale!(data,n::Int32)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    if i<=n
        @inbounds data[i]=data[i]/sqrt(Float32(n))
    end
    return
end
function study_prepare(data,scratch,n)
    scale! = ()->(@cuda threads=256 blocks=cld(n,256) study_scale!(data,Int32(n)))
    level! = len->(@cuda threads=256 blocks=cld(Int(len)÷2,256) haar_level_kernel!(data,scratch,len))
    scale!,level!
end
exit(study_main(study_prepare))

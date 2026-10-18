include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(f)
 n,o,v=f[6:8];blocks=Int(n)÷8192+1
 @cuda threads=512 blocks=blocks shmem=4*(Int(o)*Int(v)+512) scatter_kernel!(f[1],f[2],f[4],n,o,v,Val(f[9]))
end
study_main("cuda_jl",launch!)

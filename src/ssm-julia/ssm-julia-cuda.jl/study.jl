include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(f,y)
 a=f[1];batch,dim,L=f[6:8];tx=min(Int(dim),1024)
 @cuda threads=tx blocks=(Int(batch),cld(Int(dim),tx)) selective_scan_base_kernel!(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],f[2],y,batch,dim,Int32(8),L)
end
study_main("cuda_jl",launch!)

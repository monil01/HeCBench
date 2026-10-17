using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
function launch_julia!(input,out,prefix,totals,offsets,count,n,nb,nr,nc)
    nz=Int64(CUDA.count(!iszero,input))
    flat=CUDA.findall(!iszero,input)
    @cuda threads=256 blocks=cld(nz,256) write_indices_kernel!(flat,out,nr,nc,nz)
    return nz
end
include(joinpath(@__DIR__,"..","study_common.jl"));nonzero_study(launch_julia!)

using CUDA,JACC
const H_MODEL=:jacc
include(joinpath(@__DIR__,"..","hungarian_math.jl"))
function hungarian_phase!(i,v,S,nt,nb)
 tid=Int32((i-1)%nt);bid=Int32((i-1)÷nt)
 hphase!(v,tid,bid,nt,nb,S)
 nothing
end
const SPECS=Dict{Tuple{Int,Int,Int},Any}()
function launch!(v,S,nb,nt,shared)
 spec=get!(SPECS,(nb,nt,shared)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=nt,blocks=nb,shmem_size=shared,sync=false)
 end
 JACC.parallel_for(hungarian_phase!,spec,nt*nb,v,S,Int32(nt),Int32(nb))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hungarian_study(launch!)

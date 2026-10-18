using CUDA,JACC
include(joinpath(@__DIR__,"..","reduction_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function reduce_gpp!(i,j,gx,gy,args)
 tx=Int32((i-1)%32);bx=Int32((i-1)÷32);by=Int32(j-1)
 gpp_thread!(tx,bx,by,gx,gy,args...)
 nothing
end
const SPECS=Dict{Tuple{Int32,Int32},Any}()
function launch(c)
 spec=get!(SPECS,(c.gx,c.gy)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(32,1),blocks=(Int(c.gx),Int(c.gy)),shmem_size=0,sync=false)
 end
 JACC.parallel_for(reduce_gpp!,spec,(32Int(c.gx),Int(c.gy)),c.gx,c.gy,c.args)
end
gpp_study(launch)

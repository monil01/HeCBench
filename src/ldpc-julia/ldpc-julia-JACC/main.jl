using CUDA,JACC
include(joinpath(@__DIR__,"..","ldpc_math.jl"))
const LDPC_TEMPLATE=CUDA.zeros(Float32,1344)
function cnp_first!(i,j,llr,dt,R,count,compact)
 x=Int32((i-1)%96);y=Int32((j-1)%2);r=Int32((i-1)÷96);mc=Int32((j-1)÷2)
 s=cnp_prepare!(Val(true),x,y,r,mc,llr,R,count,compact,nothing)
 cnp_finish!(Val(true),r,dt,R,compact,nothing,s)
 nothing
end
function cnp_normal!(i,j,llr,dt,R,count,compact,template)
 x=Int32((i-1)%96);y=Int32((j-1)%2);r=Int32((i-1)÷96);mc=Int32((j-1)÷2);cache=JACC.shared(template)
 s=cnp_prepare!(Val(false),x,y,r,mc,llr,R,count,compact,cache)
 JACC.sync_workgroup()
 cnp_finish!(Val(false),r,dt,R,compact,cache,s)
 nothing
end
function vnp!(i,j,last,llr,dt,hd,count,compact)
 vnp_site!(last,Int32((i-1)%96),Int32((j-1)%2),Int32((i-1)÷96),Int32((j-1)÷2),llr,dt,hd,count,compact)
 nothing
end
const SPECS=Dict{Tuple{Int,Int,Int},Any}()
function spec(cols,mcw,shared)
 get!(SPECS,(cols,mcw,shared)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(96,2),blocks=(cols,mcw),shmem_size=shared,sync=false)
 end
end
function decode(c)
 for it in 1:10
  if it==1
   JACC.parallel_for(cnp_first!,spec(12,c.mcw,0),(96*12,2*c.mcw),c.llr,c.dt,c.R,c.count1,c.comp1)
  else
   JACC.parallel_for(cnp_normal!,spec(12,c.mcw,5376),(96*12,2*c.mcw),c.llr,c.dt,c.R,c.count1,c.comp1,LDPC_TEMPLATE)
  end
  JACC.parallel_for(vnp!,spec(24,c.mcw,0),(96*24,2*c.mcw),Val(it==10),c.llr,c.dt,c.hd,c.count2,c.comp2)
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
ldpc_study(decode)

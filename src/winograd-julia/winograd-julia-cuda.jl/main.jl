using CUDA
include("../study_common.jl")
function winograd_kernel(input,filter,output,offset,gx)
 lane=Int32(threadIdx().x)-1;block=Int32(blockIdx().x)-1
 ti=(block%gx)*Int32(32)+lane%Int32(32)+offset;tj=(block÷gx)*Int32(8)+lane÷Int32(32)
 store_tile!(input,filter,output,ti,tj)
 return
end
function factory(input,filter,output,cpu)
 gx=Int32(3-cpu÷32)
 gx==0 ? nothing : ()->(@cuda threads=256 blocks=gx*9 winograd_kernel(input,filter,output,Int32(cpu),gx))
end
winograd_study(factory)

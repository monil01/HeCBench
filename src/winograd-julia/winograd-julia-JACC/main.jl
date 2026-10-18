using CUDA,JACC
include("../study_portable_common.jl")
@inline function winograd_kernel(i,input,filter,output,offset,gx)
 lane=Int32((i-1)%256);block=Int32((i-1)÷256)
 ti=(block%gx)*Int32(32)+lane%Int32(32)+offset;tj=(block÷gx)*Int32(8)+lane÷Int32(32)
 store_tile!(input,filter,output,ti,tj)
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(input,filter,output,cpu)
 gx=Int32(3-cpu÷32)
 if gx==0;return nothing;end
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=Int(gx)*9,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,Int(gx)*9*256,winograd_kernel,input,filter,output,Int32(cpu),gx)
end
winograd_study(factory)

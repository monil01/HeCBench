using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel function winograd_kernel(input,filter,output,offset,gx)
 lane=Int32(@index(Local,Linear))-1;block=Int32(@index(Group,Linear))-1
 ti=(block%gx)*Int32(32)+lane%Int32(32)+offset;tj=(block÷gx)*Int32(8)+lane÷Int32(32)
 store_tile!(input,filter,output,ti,tj)
end
function factory(input,filter,output,cpu)
 gx=Int32(3-cpu÷32);k=winograd_kernel(CUDABackend(),256)
 gx==0 ? nothing : ()->k(input,filter,output,Int32(cpu),gx;ndrange=Int(gx)*9*256)
end
winograd_study(factory)

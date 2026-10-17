using CUDA, KernelAbstractions
@inline block_barrier()=KernelAbstractions.@synchronize()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function scan!(lo,hi,input,num,grid)
 tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=@localmem(Int32,(512,));scan_body!(tid,bid,grid,lo,hi,input,num,s)
end
@kernel cpu=false function finalize!(lo,hi,bits,minimum,num)
 s=@localmem(Int32,(512,));finalize_body!(Int32(@index(Local,Linear))-Int32(1),lo,hi,bits,minimum,num,s)
end
@kernel cpu=false function pack!(bits,minimum,out,input,num,grid)
 s=@localmem(UInt32,(256,));pack_body!(Int32(@index(Local,Linear))-Int32(1),Int32(@index(Group,Linear))-Int32(1),grid,bits,minimum,out,input,num,s)
end
function run!(input,num,lo,hi,bits,minimum,out,tmp,packtmp)
 backend=get_backend(input);grid=Int32(length(lo))
 scan!(backend,256)(lo,hi,input,num,grid;ndrange=256*Int(grid))
 finalize!(backend,256)(lo,hi,bits,minimum,num;ndrange=256)
 pack!(backend,256)(bits,minimum,out,input,num,grid;ndrange=256*Int(grid))
 return
end
study(run!)

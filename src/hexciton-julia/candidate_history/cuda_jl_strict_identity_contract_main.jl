using CUDA
include(joinpath(@__DIR__,"..","study_vector.jl"))
include(joinpath(@__DIR__,"..","study_generated_bodies.jl"))
include(joinpath(@__DIR__,"..","study_final_body.jl"))
function entry_comm_empty!(sin,sout,ham,dim::Int32)
 comm_empty!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_init!(sin,sout,ham,dim::Int32)
 comm_init!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_refactor!(sin,sout,ham,dim::Int32)
 comm_refactor!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_refactor_direct_store!(sin,sout,ham,dim::Int32)
 comm_refactor_direct_store!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive_constants!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive_constants!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive_constants_perm!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive_constants_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive_direct!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive_constants_direct!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive_constants_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_naive_constants_direct_perm!(sin,sout,ham,dim::Int32)
 comm_aosoa_naive_constants_direct_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa!(sin,sout,ham,dim::Int32)
 comm_aosoa!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_constants!(sin,sout,ham,dim::Int32)
 comm_aosoa_constants!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_constants_perm!(sin,sout,ham,dim::Int32)
 comm_aosoa_constants_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_direct!(sin,sout,ham,dim::Int32)
 comm_aosoa_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_constants_direct!(sin,sout,ham,dim::Int32)
 comm_aosoa_constants_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_aosoa_constants_direct_perm!(sin,sout,ham,dim::Int32)
 comm_aosoa_constants_direct_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants_perm!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants_perm_prefetch!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants_perm_prefetch!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_direct!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants_direct!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants_direct!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants_direct_prefetch!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants_direct_prefetch!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_comm_manual_aosoa_constants_direct_perm!(sin,sout,ham,dim::Int32)
 comm_manual_aosoa_constants_direct_perm!(sin,sout,ham,dim,Int32(blockIdx().x)-Int32(1),Int32(blockIdx().y)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockDim().x))
 return
end
function entry_final!(sin,sout,ham,num::Int32)
 sh=CuStaticSharedArray(Float32,490)
 final_body!(sin,sout,ham,num,Int32(blockIdx().x)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),sh,CUDA.sync_threads)
 return
end
const ENTRIES=(entry_comm_empty!,entry_comm_init!,entry_comm_refactor!,entry_comm_refactor_direct_store!,entry_comm_aosoa_naive!,entry_comm_aosoa_naive_constants!,entry_comm_aosoa_naive_constants_perm!,entry_comm_aosoa_naive_direct!,entry_comm_aosoa_naive_constants_direct!,entry_comm_aosoa_naive_constants_direct_perm!,entry_comm_aosoa!,entry_comm_aosoa_constants!,entry_comm_aosoa_constants_perm!,entry_comm_aosoa_direct!,entry_comm_aosoa_constants_direct!,entry_comm_aosoa_constants_direct_perm!,entry_comm_manual_aosoa!,entry_comm_manual_aosoa_constants!,entry_comm_manual_aosoa_constants_perm!,entry_comm_manual_aosoa_constants_perm_prefetch!,entry_comm_manual_aosoa_direct!,entry_comm_manual_aosoa_constants_direct!,entry_comm_manual_aosoa_constants_direct_prefetch!,entry_comm_manual_aosoa_constants_direct_perm!)
function launch(kid,sin,sout,ham,num)
 if kid==24
  @cuda threads=(64,2) blocks=num÷32 entry_final!(sin,sout,ham,Int32(num))
 elseif kid<10
  kernel=ENTRIES[kid+1];@cuda threads=256 blocks=num÷256 kernel(sin,sout,ham,Int32(7))
 elseif kid<16
  kernel=ENTRIES[kid+1];@cuda threads=(4,64) blocks=(1,num÷256) kernel(sin,sout,ham,Int32(7))
 else
  kernel=ENTRIES[kid+1];@cuda threads=4 blocks=num÷16 kernel(sin,sout,ham,Int32(7))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)

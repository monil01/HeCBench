using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_vector.jl"))
include(joinpath(@__DIR__,"..","study_generated_bodies.jl"))
include(joinpath(@__DIR__,"..","study_final_body.jl"))
const SHARED_TEMPLATE=JACC.zeros(Float32,490)
function entry_comm_empty!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_empty!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_init!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_init!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_refactor!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_refactor!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_refactor_direct_store!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_refactor_direct_store!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive_constants!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive_constants_perm!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive_direct!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive_constants_direct!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_naive_constants_direct_perm!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(256);BY=Int32(0);TX=q%Int32(256);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_constants!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_constants_perm!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_direct!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_constants_direct!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_aosoa_constants_direct_perm!(xx,yy,sin,sout,ham,dim::Int32)
 BX=Int32(0);BY=(Int32(yy)-Int32(1))÷Int32(64);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(64);BDX=Int32(4)
 comm_aosoa_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants_perm!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants_perm_prefetch!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_perm_prefetch!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_direct!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants_direct!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants_direct_prefetch!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct_prefetch!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_comm_manual_aosoa_constants_direct_perm!(xx,sin,sout,ham,dim::Int32)
 q=Int32(xx)-Int32(1);BX=q÷Int32(4);BY=Int32(0);TX=q%Int32(4);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
 return nothing
end
function entry_final!(xx,yy,sin,sout,ham,num::Int32,template)
 BX=(Int32(yy)-Int32(1))÷Int32(2);TX=Int32(xx)-Int32(1);TY=(Int32(yy)-Int32(1))%Int32(2)
 sh=JACC.shared(template)
 final_body!(sin,sout,ham,num,BX,TX,TY,sh,JACC.sync_workgroup)
 return nothing
end
const ENTRIES=(entry_comm_empty!,entry_comm_init!,entry_comm_refactor!,entry_comm_refactor_direct_store!,entry_comm_aosoa_naive!,entry_comm_aosoa_naive_constants!,entry_comm_aosoa_naive_constants_perm!,entry_comm_aosoa_naive_direct!,entry_comm_aosoa_naive_constants_direct!,entry_comm_aosoa_naive_constants_direct_perm!,entry_comm_aosoa!,entry_comm_aosoa_constants!,entry_comm_aosoa_constants_perm!,entry_comm_aosoa_direct!,entry_comm_aosoa_constants_direct!,entry_comm_aosoa_constants_direct_perm!,entry_comm_manual_aosoa!,entry_comm_manual_aosoa_constants!,entry_comm_manual_aosoa_constants_perm!,entry_comm_manual_aosoa_constants_perm_prefetch!,entry_comm_manual_aosoa_direct!,entry_comm_manual_aosoa_constants_direct!,entry_comm_manual_aosoa_constants_direct_prefetch!,entry_comm_manual_aosoa_constants_direct_perm!)
function launch(kid,sin,sout,ham,num)
 if kid==24
  spec=JACC.launch_spec(;threads=(64,2),blocks=(1,num÷32),shmem_size=1960,sync=false)
  JACC.parallel_for(spec,(64,num÷16),entry_final!,sin,sout,ham,Int32(num),SHARED_TEMPLATE)
 elseif kid<10
  spec=JACC.launch_spec(;threads=256,blocks=num÷256,shmem_size=0,sync=false)
  JACC.parallel_for(spec,num,ENTRIES[kid+1],sin,sout,ham,Int32(7))
 elseif kid<16
  spec=JACC.launch_spec(;threads=(4,64),blocks=(1,num÷256),shmem_size=0,sync=false)
  JACC.parallel_for(spec,(4,num÷4),ENTRIES[kid+1],sin,sout,ham,Int32(7))
 else
  spec=JACC.launch_spec(;threads=4,blocks=num÷16,shmem_size=0,sync=false)
  JACC.parallel_for(spec,num÷4,ENTRIES[kid+1],sin,sout,ham,Int32(7))
 end
 return nothing
end
include(joinpath(@__DIR__,"..","study_common_fma_secondary.jl"))
study((T,n)->JACC.zeros(T,n),launch)

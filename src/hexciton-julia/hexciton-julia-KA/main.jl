using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_vector.jl"))
include(joinpath(@__DIR__,"..","study_generated_bodies.jl"))
@kernel function entry_comm_empty!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_empty!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_init!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_init!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_refactor!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_refactor!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_refactor_direct_store!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_refactor_direct_store!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive_constants!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive_constants_perm!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive_direct!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive_constants_direct!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_naive_constants_direct_perm!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(256)
 comm_aosoa_naive_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_constants!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_constants_perm!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_direct!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_constants_direct!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_aosoa_constants_direct_perm!(sin,sout,ham,dim::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[1])-Int32(1);BY=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1);BDX=Int32(4)
 comm_aosoa_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants_perm!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants_perm_prefetch!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_perm_prefetch!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_direct!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants_direct!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants_direct_prefetch!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct_prefetch!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_comm_manual_aosoa_constants_direct_perm!(sin,sout,ham,dim::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);BY=Int32(0);TX=Int32(@index(Local,Linear))-Int32(1);TY=Int32(0);BDX=Int32(4)
 comm_manual_aosoa_constants_direct_perm!(sin,sout,ham,dim,BX,BY,TX,TY,BDX)
end
@kernel function entry_final!(sigma_in,sigma_out,ham,num::Int32)
 g=@index(Group,Cartesian);t=@index(Local,Cartesian)
 BX=Int32(g[2])-Int32(1);TX=Int32(t[1])-Int32(1);TY=Int32(t[2])-Int32(1)
 sh=@localmem(Float32,(490,))

 ij=TX;i=ij÷Int32(7);j=ij%Int32(7);sub=TY;start=BX*Int32(32)+sub*Int32(16);stop=min(num,start+Int32(16))
 if ij<Int32(49)&&sub==0
  @inbounds h=ham[ij+1];sh[ij+1]=h.data[1].value;sh[Int32(49)+ij+1]=h.data[2].value
 end
 for m in start:Int32(2):stop-Int32(1)
  @synchronize
  a=StudyV2(0f0);b=StudyV2(0f0)
  if ij<Int32(49)
   @inbounds s1=sigma_in[m*Int32(49)+ij+1];s2=sigma_in[(m+Int32(1))*Int32(49)+ij+1]
   sh[Int32(98)+sub*Int32(49)+ij+1]=s1.data[1].value;sh[Int32(294)+sub*Int32(49)+ij+1]=s1.data[2].value
   sh[Int32(98)+(Int32(2)+sub)*Int32(49)+ij+1]=s2.data[1].value;sh[Int32(294)+(Int32(2)+sub)*Int32(49)+ij+1]=s2.data[2].value
   @inbounds old1=sigma_out[m*Int32(49)+ij+1];old2=sigma_out[(m+Int32(1))*Int32(49)+ij+1]
   a=StudyV2(old1.data[1].value,old2.data[1].value);b=StudyV2(old1.data[2].value,old2.data[2].value)
  end
  @synchronize
  if ij<Int32(49)
   @inbounds for k in Int32(0):Int32(6)
    ik=i*Int32(7)+k;kj=k*Int32(7)+j
    s1=StudyV2(sh[Int32(98)+sub*Int32(49)+kj+1],sh[Int32(98)+(Int32(2)+sub)*Int32(49)+kj+1]);s2=StudyV2(sh[Int32(294)+sub*Int32(49)+kj+1],sh[Int32(294)+(Int32(2)+sub)*Int32(49)+kj+1]);a+=sh[ik+1]*s2;a+=sh[Int32(49)+ik+1]*s1;b-=sh[ik+1]*s1;b+=sh[Int32(49)+ik+1]*s2
    s1=StudyV2(sh[Int32(98)+sub*Int32(49)+ik+1],sh[Int32(98)+(Int32(2)+sub)*Int32(49)+ik+1]);s2=StudyV2(sh[Int32(294)+sub*Int32(49)+ik+1],sh[Int32(294)+(Int32(2)+sub)*Int32(49)+ik+1]);a-=sh[kj+1]*s2;a+=sh[Int32(49)+kj+1]*s1;b+=sh[kj+1]*s1;b-=sh[Int32(49)+kj+1]*s2
   end
   @inbounds sigma_out[m*Int32(49)+ij+1]=StudyV2(a.data[1].value,b.data[1].value);sigma_out[(m+Int32(1))*Int32(49)+ij+1]=StudyV2(a.data[2].value,b.data[2].value)
  end
 end
end
const ENTRIES=(entry_comm_empty!,entry_comm_init!,entry_comm_refactor!,entry_comm_refactor_direct_store!,entry_comm_aosoa_naive!,entry_comm_aosoa_naive_constants!,entry_comm_aosoa_naive_constants_perm!,entry_comm_aosoa_naive_direct!,entry_comm_aosoa_naive_constants_direct!,entry_comm_aosoa_naive_constants_direct_perm!,entry_comm_aosoa!,entry_comm_aosoa_constants!,entry_comm_aosoa_constants_perm!,entry_comm_aosoa_direct!,entry_comm_aosoa_constants_direct!,entry_comm_aosoa_constants_direct_perm!,entry_comm_manual_aosoa!,entry_comm_manual_aosoa_constants!,entry_comm_manual_aosoa_constants_perm!,entry_comm_manual_aosoa_constants_perm_prefetch!,entry_comm_manual_aosoa_direct!,entry_comm_manual_aosoa_constants_direct!,entry_comm_manual_aosoa_constants_direct_prefetch!,entry_comm_manual_aosoa_constants_direct_perm!)
function launch(kid,sin,sout,ham,num)
 if kid==24
  entry_final!(BACKEND,(64,2))(sin,sout,ham,Int32(num);ndrange=(64,2*(num÷32)))
 elseif kid<10
  ENTRIES[kid+1](BACKEND,256)(sin,sout,ham,Int32(7);ndrange=num)
 elseif kid<16
  ENTRIES[kid+1](BACKEND,(4,64))(sin,sout,ham,Int32(7);ndrange=(4,64*(num÷256)))
 else
  ENTRIES[kid+1](BACKEND,4)(sin,sout,ham,Int32(7);ndrange=num÷4)
 end
end
include(joinpath(@__DIR__,"..","study_common_fma_secondary.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)

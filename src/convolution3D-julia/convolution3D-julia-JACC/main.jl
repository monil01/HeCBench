using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,j,k,x,w,y,C,M,K,Hin,Win,Hout,Wout,Wgrid,mode)
 ii=Int32(i)-Int32(1);jj=Int32(j)-Int32(1);kk=Int32(k)-Int32(1)
 conv_body(mod(ii,Int32(16)),mod(jj,Int32(16)),ii÷Int32(16),jj÷Int32(16),kk,x,w,y,C,M,K,Hin,Win,Hout,Wout,Wgrid,mode)
 return
end
function run!(x,w,y,dims,::Val{STYLE}) where STYLE
 g=grid(STYLE);JACC.parallel_for(JACC.launch_spec(threads=(16,16,1),blocks=g,shmem_size=0,sync=false),(16g[1],16g[2],g[3]),kernel!,x,w,y,dims...,Val(STYLE))
 return
end
study(run!)

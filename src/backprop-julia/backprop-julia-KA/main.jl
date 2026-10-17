using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function layerforward_ka!(input,weights,partial)
 x,y=@index(Local,NTuple);_,by=@index(Group,NTuple)
 inode=@localmem Float32 (16,)
 tile=@localmem Float32 (256,)
 gy=(by-1)*16+y;idx=gy*17+x+1;ti=(y-1)*16+x
 if x==1;inode[y]=input[gy+1];end
 @synchronize
 tile[ti]=weights[idx]
 @synchronize
 tile[ti]*=inode[y]
 @synchronize
 power=2
 while power<=16
  if (y-1)%power==0;tile[ti]+=tile[ti+(power÷2)*16];end
  @synchronize
  power*=2
 end
 weights[idx]=tile[ti]
 @synchronize
 if x==1;partial[(by-1)*16+y]=tile[y];end
end
@kernel function adjust_ka!(input,weights,delta,prev)
 x,y=@index(Local,NTuple);_,by=@index(Group,NTuple);gy=(by-1)*16+y;idx=gy*17+x+1
 u=ETA*delta[x+1]*input[gy+1]+MOMENTUM*prev[idx]
 weights[idx]+=u;prev[idx]=u
 @synchronize
 if y==1 && by==1
  bi=x+1;v=ETA*delta[bi]+MOMENTUM*prev[bi];weights[bi]+=v;prev[bi]=v
 end
end
function forward!(a,w,p,n)
 layerforward_ka!(CUDABackend(),(16,16))(a,w,p;ndrange=(16,n))
end
function adjust!(a,w,d,p,n)
 adjust_ka!(CUDABackend(),(16,16))(a,w,d,p;ndrange=(16,n))
end
study_main(net->study_train_tiled!(net,forward!,adjust!))

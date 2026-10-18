using CUDA,KernelAbstractions,Atomix
include("device.jl")
@kernel unsafe_indices=true function entry_0(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(0),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_1(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(1),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_2(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(2),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_3(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Int32,(S,))
 snicit_body(Val(3),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_4(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(4),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_5(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(5),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_6(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(6),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_7(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(7),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
@kernel unsafe_indices=true function entry_8(Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C, bx::Int32,by::Int32,::Val{S}) where S
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem(Float32,(S,))
 snicit_body(Val(8),Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C,tid%bx,tid÷bx,bx,by,row,scratch)
end
function launch(op,args,a,stream)
 f=getfield(@__MODULE__,Symbol(:entry_,op));threads=Int(a[6]*a[7]);groups=Int(a[5]);S=max(Int(a[8])÷4,1)
 kernel=f(CUDABackend(),threads)
 CUDA.stream!(stream) do
  kernel(args...,Int32(a[6]),Int32(a[7]),Val(S);ndrange=threads*groups)
 end
 nothing
end
include("../study_common.jl")
main()

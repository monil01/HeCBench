@inline function fragment_combine(acc::F,cf::F)where F
 F(ntuple(i->muladd(0.5f0,acc.x[i],2f0*cf.x[i]),Val(8)))
end
# True warp-cooperative CUDA.WMMA tensor operations, not a library GEMM substitute.
@inline function wmma_tile(A,B,C,D,n,k,row,col,::Val{IMPL})where IMPL
 conf=CUDA.WMMA.Config{16,16,16,Float32}
 acc=CUDA.WMMA.fill_c(0f0,conf)
 cf=CUDA.WMMA.fill_c(0f0,conf)
 h=Int32(0)
 while h<k
  af=CUDA.WMMA.load_a(pointer(A,row*k+h+Int32(1)),k,CUDA.WMMA.RowMajor,conf)
  bf=CUDA.WMMA.load_b(pointer(B,col*k+h+Int32(1)),k,CUDA.WMMA.ColMajor,conf)
  acc=CUDA.WMMA.mma(af,bf,acc,conf)
  if IMPL==0
   cf=CUDA.WMMA.load_c(pointer(C,row*n+col+Int32(1)),n,CUDA.WMMA.RowMajor,conf)
   cf=fragment_combine(acc,cf)
  end
  h+=Int32(16)
 end
 if IMPL==1
  cf=CUDA.WMMA.load_c(pointer(C,row*n+col+Int32(1)),n,CUDA.WMMA.RowMajor,conf)
  cf=fragment_combine(acc,cf)
 end
 CUDA.WMMA.store_d(pointer(D,row*n+col+Int32(1)),cf,n,CUDA.WMMA.RowMajor,conf)
 nothing
end

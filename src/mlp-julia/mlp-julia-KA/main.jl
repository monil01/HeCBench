using CUDA, KernelAbstractions
include("../study_common.jl")
@kernel function dense_tile!(y,x,w,bias,ifeat,ofeat,use_bias,use_relu)
 tid=@index(Local,Linear)-1
 block=@index(Group,Linear)-1
 lx=tid%16; ly=tid÷16
 tilesm=cld(ofeat,16)
 mb=(block%tilesm)*16; nb=(block÷tilesm)*16
 tile=@localmem Float32 (512,)
 acc=0f0
 for kb in 0:16:ifeat-1
  k=kb+lx
  mw=mb+ly; nx=nb+ly
  @inbounds tile[tid+1]=(mw<ofeat && k<ifeat) ? w[k+ifeat*mw+1] : 0f0
  @inbounds tile[257+tid]=(nx<128 && k<ifeat) ? x[k+ifeat*nx+1] : 0f0
  @synchronize
  for kk in 0:15
   @inbounds acc=muladd(tile[kk+16*lx+1],tile[257+kk+16*ly],acc)
  end
  @synchronize
 end
 m=mb+lx; n=nb+ly
 if m<ofeat && n<128
  if use_bias; @inbounds acc+=bias[m+1]; end
  if use_relu; acc=max(acc,0f0); end
  @inbounds y[m+ofeat*n+1]=acc
 end
end
function mlp_ka_factory(x,weights,biases,outputs,use_bias,use_relu)
 kernel=dense_tile!(CUDABackend(),256)
 op=()->begin
  input=x
  for (layer,(ifeat,ofeat)) in enumerate(((784,512),(512,512),(512,10)))
   kernel(outputs[layer],input,weights[layer],biases[layer],ifeat,ofeat,use_bias,use_relu;ndrange=cld(ofeat,16)*8*256)
   input=outputs[layer]
  end
 end
 (op,()->nothing)
end
main()=mlp_study(mlp_ka_factory)
if abspath(PROGRAM_FILE)==@__FILE__; main(); end

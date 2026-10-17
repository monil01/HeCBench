using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function dense_tile!(index,y,x,w,bias,ifeat,ofeat,use_bias,use_relu,template)
 tid=(index-1)%256
 block=(index-1)÷256
 lx=tid%16; ly=tid÷16
 tilesm=cld(ofeat,16)
 mb=(block%tilesm)*16; nb=(block÷tilesm)*16
 tile=JACC.shared(template)
 acc=0f0
 for kb in 0:16:ifeat-1
  k=kb+lx
  mw=mb+ly; nx=nb+ly
  @inbounds tile[tid+1]=(mw<ofeat && k<ifeat) ? w[k+ifeat*mw+1] : 0f0
  @inbounds tile[257+tid]=(nx<128 && k<ifeat) ? x[k+ifeat*nx+1] : 0f0
  JACC.sync_workgroup()
  for kk in 0:15
   @inbounds acc=muladd(tile[kk+16*lx+1],tile[257+kk+16*ly],acc)
  end
  JACC.sync_workgroup()
 end
 m=mb+lx; n=nb+ly
 if m<ofeat && n<128
  if use_bias; @inbounds acc+=bias[m+1]; end
  if use_relu; acc=max(acc,0f0); end
  @inbounds y[m+ofeat*n+1]=acc
 end
end
function mlp_jacc_factory(x,weights,biases,outputs,use_bias,use_relu)
 template=CUDA.zeros(Float32,512)
 specs=[JACC.launch_spec(;threads=256,blocks=cld(n,16)*8,shmem_size=2048,sync=false) for n in (512,512,10)]
 op=()->begin
  input=x
  for (layer,(ifeat,ofeat)) in enumerate(((784,512),(512,512),(512,10)))
   JACC.parallel_for(specs[layer],cld(ofeat,16)*8*256,dense_tile!,outputs[layer],input,weights[layer],biases[layer],ifeat,ofeat,use_bias,use_relu,template)
   input=outputs[layer]
  end
 end
 (op,()->nothing)
end
main()=mlp_study(mlp_jacc_factory)
if abspath(PROGRAM_FILE)==@__FILE__; main(); end

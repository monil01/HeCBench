using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel unsafe_indices=true function conv_basic_ka!(input,output,mask,N::Int32,MW::Int32)
 i=Int32(@index(Global,Linear))-Int32(1);start=i-MW÷Int32(2);acc=zero(eltype(output))
 for j in Int32(0):(MW-Int32(1))
  idx=start+j
  if 0<=idx<N;@inbounds acc+=input[idx+Int32(1)]*mask[j+Int32(1)];end
 end
 @inbounds output[i+Int32(1)]=acc
end
@kernel unsafe_indices=true function conv_tiled_ka!(input::AbstractVector{T},output,mask,N::Int32,MW::Int32,::Val{MODE},::Val{BT}) where {T,MODE,BT}
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1);bt=Int32(BT);i=block*bt+tid;n=MW÷Int32(2)
 tile=@localmem T (BT+(MODE==2 ? 9 : 0),)
 if MODE==2
  left=(block-Int32(1))*bt+tid
  if tid>=bt-n;@inbounds tile[tid-(bt-n)+Int32(1)]=left<0 ? zero(T) : input[left+Int32(1)];end
  @inbounds tile[n+tid+Int32(1)]=input[i+Int32(1)]
  right=(block+Int32(1))*bt+tid
  if tid<n;@inbounds tile[tid+bt+n+Int32(1)]=right>=N ? zero(T) : input[right+Int32(1)];end
 else
  @inbounds tile[tid+Int32(1)]=input[i+Int32(1)]
 end
 @synchronize
 acc=zero(T)
 for j in Int32(0):(MW-Int32(1))
  if MODE==2
   @inbounds acc+=tile[tid+j+Int32(1)]*mask[j+Int32(1)]
  else
   idx=i-n+j
   if 0<=idx<N
    @inbounds acc+=(block*bt<=idx<(block+Int32(1))*bt ? tile[tid+j-n+Int32(1)] : input[idx+Int32(1)])*mask[j+Int32(1)]
   end
  end
 end
 @inbounds output[i+Int32(1)]=acc
end
function study_launch!(a,o,m,mw,mode::Val{MODE},bt::Val{BT}) where {MODE,BT}
 if MODE==1
  conv_basic_ka!(CUDABackend(),BT)(a,o,m,Int32(2048),mw;ndrange=2048)
 else
  conv_tiled_ka!(CUDABackend(),BT)(a,o,m,Int32(2048),mw,mode,bt;ndrange=2048)
 end
end
study_main(study_launch!)

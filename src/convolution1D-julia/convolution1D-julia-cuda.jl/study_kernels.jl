function conv_tiled_kernel!(input,output,mask,N::Int32,MW::Int32,::Val{MODE}) where {MODE}
 tid=threadIdx().x-Int32(1);bt=blockDim().x;block=blockIdx().x-Int32(1);i=block*bt+tid;n=MW÷Int32(2)
 tile=CUDA.@cuDynamicSharedMem(eltype(output),Int(bt+(MODE==2 ? Int32(9) : Int32(0))))
 if MODE==2
  left=(block-Int32(1))*bt+tid
  if tid>=bt-n;@inbounds tile[tid-(bt-n)+Int32(1)]=left<0 ? zero(eltype(output)) : input[left+Int32(1)];end
  @inbounds tile[n+tid+Int32(1)]=input[i+Int32(1)]
  right=(block+Int32(1))*bt+tid
  if tid<n;@inbounds tile[tid+bt+n+Int32(1)]=right>=N ? zero(eltype(output)) : input[right+Int32(1)];end
 else
  @inbounds tile[tid+Int32(1)]=input[i+Int32(1)]
 end
 CUDA.sync_threads()
 acc=zero(eltype(output))
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
 return
end

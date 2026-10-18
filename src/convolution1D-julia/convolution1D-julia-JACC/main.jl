using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const CONV_BACKEND=JACC.get_backend(:cuda)
const CONV_SPECS=Dict((T,BT,MODE)=>JACC.LaunchSpec{typeof(CONV_BACKEND)}(;threads=BT,blocks=2048÷BT,shmem_size=(MODE==1 ? 0 : (BT+(MODE==2 ? 9 : 0))*sizeof(T)),sync=false) for T in (Float64,Float32,Int16),BT in CONV_BTS,MODE in 1:3)
const CONV_TEMPLATES=Dict((T,BT,MODE)=>CUDA.zeros(T,BT+(MODE==2 ? 9 : 0)) for T in (Float64,Float32,Int16),BT in CONV_BTS,MODE in 2:3)
function conv_jacc!(global_i,input,output,mask,N::Int32,MW::Int32,::Val{MODE},::Val{BT},template) where {MODE,BT}
 i=Int32(global_i)-Int32(1);bt=Int32(BT);tid=i%bt;block=i÷bt;n=MW÷Int32(2);acc=zero(eltype(output))
 if MODE==1
  for j in Int32(0):(MW-Int32(1))
   idx=i-n+j
   if 0<=idx<N;@inbounds acc+=input[idx+Int32(1)]*mask[j+Int32(1)];end
  end
 else
  tile=JACC.shared(template)
  if MODE==2
   left=(block-Int32(1))*bt+tid
   if tid>=bt-n;@inbounds tile[tid-(bt-n)+Int32(1)]=left<0 ? zero(eltype(output)) : input[left+Int32(1)];end
   @inbounds tile[n+tid+Int32(1)]=input[i+Int32(1)]
   right=(block+Int32(1))*bt+tid
   if tid<n;@inbounds tile[tid+bt+n+Int32(1)]=right>=N ? zero(eltype(output)) : input[right+Int32(1)];end
  else
   @inbounds tile[tid+Int32(1)]=input[i+Int32(1)]
  end
  JACC.sync_workgroup()
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
 end
 @inbounds output[i+Int32(1)]=acc
 nothing
end
function study_launch!(a,o,m,mw,mode::Val{MODE},bt::Val{BT}) where {MODE,BT}
 key=(eltype(a),BT,MODE);template=MODE==1 ? nothing : CONV_TEMPLATES[key]
 JACC.parallel_for(conv_jacc!,CONV_SPECS[key],Int32(2048),a,o,m,Int32(2048),mw,mode,bt,template)
end
study_main(study_launch!)

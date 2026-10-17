using Base64
const RUNTIME_BASE64=CUDA.CUDACore.CUDA_Runtime.libcudart
const TABLE64=Tuple(codeunits("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"))
@inline function encode_body(id,input,output,pad)
 if id<Int32(1024)
  @inbounds a,b,c=input[id+1]
  last=pad!=0&&id==Int32(1023);x=Int32(a);y=Int32(b);z=Int32(c)
  v1=TABLE64[(x>>2)+1];v2=TABLE64[((x&3)<<4)+(last&&pad==1 ? 0 : y>>4)+1];v3=last ? (pad==1 ? UInt8('=') : TABLE64[((y&15)<<2)+1]) : TABLE64[((y&15)<<2)+(z>>6)+1];v4=last ? UInt8('=') : TABLE64[(z&63)+1]
  @inbounds output[id+1]=(v1,v2,v3,v4)
 end
 return
end
function rtcheck(status)
 status==0||error("CUDA runtime call failed ",status)
end
function rtcopy(dst,src,bytes,kind,stream,asynchronous)
 if asynchronous
  rtcheck(ccall((:cudaMemcpyAsync,RUNTIME_BASE64),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),dst,src,bytes,kind,Ptr{Cvoid}(stream.handle)))
 else
  rtcheck(ccall((:cudaMemcpy,RUNTIME_BASE64),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),dst,src,bytes,kind))
 end
end
function study(run!)
 input=CuArray{NTuple{3,UInt8}}(undef,1024);output=CuArray{NTuple{4,UInt8}}(undef,1024);streams=[CUDA.CuStream() for _ in 1:4];single=CUDA.legacy_stream();hi=Ref{Ptr{Cvoid}}();ho=Ref{Ptr{Cvoid}}();rtcheck(ccall((:cudaMallocHost,RUNTIME_BASE64),Cint,(Ref{Ptr{Cvoid}},Csize_t),hi,3072));rtcheck(ccall((:cudaMallocHost,RUNTIME_BASE64),Cint,(Ref{Ptr{Cvoid}},Csize_t),ho,4096));hostin=unsafe_wrap(Vector{UInt8},Ptr{UInt8}(hi[]),3072;own=false);hostout=unsafe_wrap(Vector{UInt8},Ptr{UInt8}(ho[]),4096;own=false);dp=Ptr{Cvoid}(UInt(pointer(input)));op=Ptr{Cvoid}(UInt(pointer(output)))
 function sequence!(pad,multi)
  if multi
   for part in 0:3
    st=streams[part+1];rtcopy(dp+768*part,hi[]+768*part,768,1,st,true);run!(input,output,Int32(pad),Int32(256*part),256,st);rtcopy(ho[]+1024*part,op+1024*part,1024,2,st,true)
   end
   for st in streams;CUDA.synchronize(st);end
  else
   rtcopy(dp,hi[],3072,1,single,false);run!(input,output,Int32(pad),Int32(0),1024,single);rtcopy(ho[],op,4096,2,single,false)
  end
 end
 for pad in 0:2,multi in (false,true)
  n=pad==0 ? 3072 : 3069+pad;fill!(hostin,0x00);for i in 0:n-1;hostin[i+1]=UInt8(mod(37*i+11,256));end;sequence!(pad,multi)
 end
 CUDA.synchronize()
 for trial in 1:5
  total=0.;got=UInt8[];want=UInt8[]
  for pad in 0:2,multi in (false,true)
   n=pad==0 ? 3072 : 3069+pad;fill!(hostin,0x00);for i in 0:n-1;hostin[i+1]=UInt8(mod(37*i+11,256));end;expected=collect(codeunits(base64encode(hostin[1:n])));CUDA.synchronize();t=time_ns();for repeat in 1:10;sequence!(pad,multi);end;total+=(time_ns()-t)*1e-9/10;append!(got,hostout);append!(want,expected)
  end
  ok=got==want;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":24576,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"sum_three_padding_one_four_stream_transfer_encode_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint8\",\"count\":24576,\"byte_order\":\"little\"}}")
  ok||error("Base64standardlibrary fullbytes failed")
 end
 rtcheck(ccall((:cudaFreeHost,RUNTIME_BASE64),Cint,(Ptr{Cvoid},),hi[]));rtcheck(ccall((:cudaFreeHost,RUNTIME_BASE64),Cint,(Ptr{Cvoid},),ho[]))
end

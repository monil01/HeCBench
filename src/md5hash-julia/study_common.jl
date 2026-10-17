using CUDA,Printf,Libdl
function openssl_md5(words,len)
 bytes=collect(reinterpret(UInt8,UInt32[words[1],words[2]]));out=Vector{UInt8}(undef,16)
 ccall((:MD5,"libcrypto.so.3"),Ptr{UInt8},(Ptr{UInt8},Csize_t,Ptr{UInt8}),bytes,len,out)==C_NULL&&error("CPUOpenSSLMD5failed")
 Tuple(reinterpret(UInt32,out))
end
function make_fixture()
 fs=[];pw0=UInt32[];pw1=UInt32[];pl=Int32[];pg=UInt32[]
 for (id,(len,base)) in enumerate(((7,4),(5,7),(6,5),(5,8)))
  n=base^len;target=id==1 ? 0 : id==2 ? n-1 : id==3 ? n÷3+7 : -1;words=target<0 ? (UInt32(0),UInt32(0)) : study_words(Int32(target),Int32(base));sd=target<0 ? (UInt32(0),UInt32(0),UInt32(0),UInt32(0)) : openssl_md5(words,len);key=target<0 ? zeros(UInt8,8) : collect(reinterpret(UInt8,UInt32[words...]));matches=0
  for i in 0:n-1
   w0,w1=study_words(Int32(i),Int32(base));gold=openssl_md5((w0,w1),len);matches+=gold==sd;push!(pw0,w0);push!(pw1,w1);push!(pl,Int32(len));append!(pg,gold)
  end
  @assert matches==(target<0 ? 0 : 1)
  push!(fs,(len=Int32(len),base=Int32(base),n=Int32(n),sd=sd,target=Int32(target),key=key,expected_digest=target<0 ? zeros(UInt32,4) : UInt32[sd...]))
 end
 # Allpaddingpositions0..7 andhighbitbytes. Empty,a,abc also havefixedknown standarddigests.
 for len in 0:7
  bytes=zeros(UInt8,8);for i in 1:len;bytes[i]=UInt8(96+i);end;w=Tuple(reinterpret(UInt32,bytes));z=openssl_md5(w,len)
  if len in (0,1,3)
   expected=len==0 ? "d41d8cd98f00b204e9800998ecf8427e" : len==1 ? "0cc175b9c0f1b6a831c399e269772661" : "900150983cd24fb0d6963f7d28e17f72";@assert bytes2hex(collect(reinterpret(UInt8,UInt32[z...])))==expected
  end
  @assert study_md5(w[1],w[2],Int32(len))==z
  push!(pw0,w[1]);push!(pw1,w[2]);push!(pl,Int32(len));append!(pg,z)
 end
 for len in 1:7
  bytes=zeros(UInt8,8);for i in 1:len;bytes[i]=UInt8(0x80+i*7);end;w=Tuple(reinterpret(UInt32,bytes));z=openssl_md5(w,len);push!(pw0,w[1]);push!(pw1,w[2]);push!(pl,Int32(len));append!(pg,z)
 end
 prefix=Int64[]
 for f in fs;append!(prefix,Int64[f.len,f.base,f.n,f.sd...,-1,zeros(Int64,12)...]);end
 append!(prefix,Int64[length(pl)]);append!(prefix,Int64.(pw0));append!(prefix,Int64.(pw1));append!(prefix,Int64.(pl))
 (fs,pw0,pw1,pl,pg,prefix)
end
function study_main(search_call!,probe_launch!)
 rep=isempty(ARGS) ? 5 : parse(Int,ARGS[1]);fs,pw0,pw1,pl,gold,prefix=make_fixture();outputs=[(fill(Int32(-1),1),zeros(UInt8,8),zeros(UInt32,4)) for f in fs];d0=CuArray(pw0);d1=CuArray(pw1);dl=CuArray(pl);dop=CuArray{UInt32}(undef,length(gold))
 ver=unsafe_string(ccall((:OpenSSL_version,"libcrypto.so.3"),Cstring,(Cint,),0));println("CPU_ORACLE_OPENSSL ",ver," paths=",filter(p->occursin("libcrypto",p),Libdl.dllist()))
 for (f,o) in zip(fs,outputs);search_call!(f,o...);end;probe_launch!(d0,d1,dl,dop,Int32(length(pl)));CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();st=time_ns();for _ in 1:rep,(f,o) in zip(fs,outputs);search_call!(f,o...);end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep
  # Untimed all-candidate digestverification. Doesnotcontribute tosearchlatency.
  probe_launch!(d0,d1,dl,dop,Int32(length(pl)));CUDA.synchronize();pg=Array(dop);ok=pg==gold;got=Int64[]
  for (f,o) in zip(fs,outputs)
   ok&=o[1][1]==f.target&&o[2]==f.key&&o[3]==f.expected_digest;append!(got,Int64.(o[1]));append!(got,Int64.(o[2]));append!(got,Int64.(o[3]))
  end
  append!(got,Int64.(pg));art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"four_complete_MD5searches_hostinclusive_alloc_initcopy_kernel_copy_free_excludes_digestprobe\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),rep,path,length(art));ok||error("fullsearch+untimedallcandidateMD5digestcontract")
 end
end

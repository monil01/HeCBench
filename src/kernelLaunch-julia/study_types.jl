struct KLArgs{N}
 bytes::NTuple{N,UInt8}
end
@inline function kl_store(args::KLArgs, out::Core.LLVMPtr{UInt8,1}, i)
 if reinterpret(UInt64,out) != UInt64(0)
  @inbounds unsafe_store!(out,args.bytes[Int(i)+1])
 end
 return
end

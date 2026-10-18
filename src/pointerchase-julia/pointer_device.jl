# Ordinary nonvolatile 64-bit global-memory load via LLVM backend extension.
# Exact native pointer representation; no UInt32 index substitution.
@inline function pointer_next(p::UInt64)
 Base.llvmcall(("define i64 @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace(1)\n %v = load i64, ptr addrspace(1) %p, align 8\n ret i64 %v }", "entry"),UInt64,Tuple{UInt64},p)
end
@inline function chase_pointer(p::UInt64, accesses::UInt32)
 i=UInt32(0)
 while i < accesses
  p=pointer_next(p);i+=UInt32(1)
 end
 p
end

using CUDA
@inline function sys_add(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.add.u32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_exch(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.exch.b32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_max(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.max.s32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_min(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.min.s32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_and(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.and.b32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_or(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.or.b32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_xor(p::Core.LLVMPtr{Int32,A},v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.xor.b32 \$0, [\$1], \$2;", "=r,l,r", true, Int32, Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
end
@inline function sys_cas(p::Core.LLVMPtr{Int32,A},cmp::Int32,v::Int32) where A
 CUDA.CUDACore.@asmcall("atom.relaxed.sys.global.cas.b32 \$0, [\$1], \$2, \$3;", "=r,l,r,r",true,Int32,Tuple{Core.LLVMPtr{Int32,A},Int32,Int32},p,cmp,v)
end
@inline function system_lane!(a,tid::Int32,loops::Int32)
 for j in Int32(1):loops
  sys_add(pointer(a,17),Int32(10));sys_exch(pointer(a,18),tid);sys_max(pointer(a,19),tid);sys_min(pointer(a,20),tid)
  sys_cas(pointer(a,23),tid-Int32(1),tid);sys_and(pointer(a,24),Int32(2)*tid+Int32(7));sys_or(pointer(a,25),Int32(1)<<tid);sys_xor(pointer(a,26),tid)
 end
 nothing
end

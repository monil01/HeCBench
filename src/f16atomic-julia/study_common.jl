using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
# CUDA-backend extension preserves the source packed two-half atomic instruction.
@inline function packed_half2_add!(ptr::Core.LLVMPtr{UInt32,1})
 CUDA.CUDACore.@asmcall("atom.global.add.noftz.f16x2 \$0, [\$1], \$2;", "=r,l,r", true, UInt32, Tuple{Core.LLVMPtr{UInt32,1},UInt32},ptr,UInt32(0x3c000000))
 return
end
function f16atomic_study(factory)
 nelems=32768;ntids=div(nelems,2);repeats=10
 expected=Float32[isodd(i) ? 0f0 : Float32(div(ntids,256)*repeats) for i in 1:512]
 output=CUDA.zeros(UInt32,256);launch! = factory(output,ntids)
 readback=()->Float32.(collect(reinterpret(Float16,Array(output))))
 OwnedStudy.run_trials!(launch!,()->fill!(output,UInt32(0)),readback,expected,repeats;atol=0.,rtol=0.)
end

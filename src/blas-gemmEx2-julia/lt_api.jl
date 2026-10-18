using CUDA,Printf,Libdl
const LT_LIB="/usr/lib/x86_64-linux-gnu/libcublasLt.so.12.0.2.224"
const RT_LIB="/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146"
struct LtAlgo
 data::NTuple{8,UInt64}
end
struct LtHeuristic
 algo::LtAlgo
 workspace::Csize_t
 state::Cint
 waves::Float32
 reserved::NTuple{4,Cint}
end
@assert sizeof(LtAlgo)==64&&sizeof(LtHeuristic)==96&&fieldoffset(LtHeuristic,3)==72&&fieldoffset(LtHeuristic,4)==76
ltck(x)=x==0 ? nothing : error("CUBLASLT status $(x)")
rtck(x)=x==0 ? nothing : error("CUDART status $(x)")
function raw_dtype(code)
 code==0 ? (Float64,Float64,Float64,Int32(1),Int32(1),Int32(1),Int32(70)) : code<=4 ? (Float32,Float32,Float32,Int32(0),Int32(0),Int32(0),Int32(68)) : code==5 ? (Float16,Float16,Float16,Int32(2),Int32(2),Int32(2),Int32(64)) : code==6 ? (Float16,Float16,Float32,Int32(2),Int32(2),Int32(0),Int32(68)) : code==7 ? (UInt16,UInt16,Float32,Int32(14),Int32(14),Int32(0),Int32(68)) : (Int8,Int32,Int32,Int32(3),Int32(10),Int32(10),Int32(72))
end
encode_bf(x)=UInt16(reinterpret(UInt32,Float32(x))>>16)
decode_bf(x::UInt16)=Float64(reinterpret(Float32,UInt32(x)<<16))
function lt_pipeline(c,handle,workspace)
 op=Ref{Ptr{Cvoid}}();pref=Ref{Ptr{Cvoid}}();la=Ref{Ptr{Cvoid}}();lb=Ref{Ptr{Cvoid}}();lc=Ref{Ptr{Cvoid}}();none=Ref{Cint}(0);sz=Ref{Csize_t}(33554432)
 ltck(ccall((:cublasLtMatmulDescCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},Cint,Cint),op,c.compute,c.scale))
 for attr in (3,4);ltck(ccall((:cublasLtMatmulDescSetAttribute,LT_LIB),Cint,(Ptr{Cvoid},Cint,Ref{Cint},Csize_t),op[],attr,none,4));end
 ltck(ccall((:cublasLtMatrixLayoutCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},Cint,UInt64,UInt64,Int64),la,c.atype,c.n,c.k,c.n));ltck(ccall((:cublasLtMatrixLayoutCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},Cint,UInt64,UInt64,Int64),lb,c.atype,c.k,c.m,c.k));ltck(ccall((:cublasLtMatrixLayoutCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},Cint,UInt64,UInt64,Int64),lc,c.ctype,c.n,c.m,c.n));ltck(ccall((:cublasLtMatmulPreferenceCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},),pref));ltck(ccall((:cublasLtMatmulPreferenceSetAttribute,LT_LIB),Cint,(Ptr{Cvoid},Cint,Ref{Csize_t},Csize_t),pref[],1,sz,sizeof(Csize_t)))
 heuristic=Ref{LtHeuristic}();count=Ref{Cint}(0);ltck(ccall((:cublasLtMatmulAlgoGetHeuristic,LT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Ref{LtHeuristic},Ref{Cint}),handle,op[],la[],lb[],lc[],lc[],pref[],1,heuristic,count));@assert count[]==1&&heuristic[].state==0;algo=Ref(heuristic[].algo)
 GC.@preserve c begin
  ltck(ccall((:cublasLtMatmul,LT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ref{LtAlgo},Ptr{Cvoid},Csize_t,Ptr{Cvoid}),handle,op[],pointer(c.alpha),c.B.d,la[],c.A.d,lb[],pointer(c.beta),c.C.d,lc[],c.C.d,lc[],algo,workspace,33554432,C_NULL))
 end
 rtck(ccall((:cudaDeviceSynchronize,RT_LIB),Cint,()))
 ltck(ccall((:cublasLtMatmulPreferenceDestroy,LT_LIB),Cint,(Ptr{Cvoid},),pref[]));ltck(ccall((:cublasLtMatmulDescDestroy,LT_LIB),Cint,(Ptr{Cvoid},),op[]));for layout in (la[],lb[],lc[]);ltck(ccall((:cublasLtMatrixLayoutDestroy,LT_LIB),Cint,(Ptr{Cvoid},),layout));end
 nothing
end

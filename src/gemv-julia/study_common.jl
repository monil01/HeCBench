using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
@inline q8(mat,row,col,n)=Float32(@inbounds mat[row*n+col+1])
@inline function q4(mat,row,col,n)
 byte=@inbounds mat[row*div(n,Int32(2))+div(col,Int32(2))+1]
 Float32(iseven(col) ? byte&UInt8(0x0f) : (byte>>4)&UInt8(0x0f))
end
function gemv_study(factory)
 packed=!isempty(ARGS)&&ARGS[1]=="int4";n=512
 matrix=packed ? UInt8[UInt8(mod(row*3+j*7+div(row,5),16))|(UInt8(mod(row*11+j*5+div(j,3),16))<<4) for row in 0:n-1 for j in 0:div(n,2)-1] : Int8[mod(row*17+col*23+div(row,7),256)-128 for row in 0:n-1 for col in 0:n-1]
 vector=Float16[(mod(i*19,127)+1)/128 for i in 0:n-1];scale=Float16(0.0625);zp=Float16(0.01);q=packed ? q4 : q8;chunk=packed ? 16 : 8
 expected=zeros(Float32,n);direct=zeros(Float64,n)
 for row in 0:n-1
  lane=zeros(Float32,32)
  for tid in 0:31
   acc=0.f0
   for iteration in 0:div(n,32*chunk)-1
    base=(tid+iteration*32)*chunk
    for off in 0:chunk-1
     term=q(matrix,Int32(row),Int32(base+off),Int32(n))-Float32(zp)
     acc=fma(Float32(vector[base+off+1]),term,acc)
    end
   end
   lane[tid+1]=acc*Float32(scale)
  end
  for stride in (16,8,4,2,1)
   for tid in 1:stride;lane[tid]+=lane[tid+stride];end
  end
  expected[row+1]=Float32(Float16(lane[1]))
  direct[row+1]=sum((Float64(q(matrix,Int32(row),Int32(col),Int32(n)))-Float64(zp))*Float64(scale)*Float64(vector[col+1]) for col in 0:n-1)
 end
 maximum(abs.(Float64.(expected)-direct))<=.125||error("emulator violates original absolute tolerance")
 mat=CuArray(matrix);vec=CuArray(vector);res=CUDA.zeros(Float16,n);operation! =factory(mat,vec,res,n,scale,zp,packed)
 readback=()->begin
  got=Float32.(Array(res));delta=maximum(abs.(Float64.(got)-direct));println("STUDY_DIRECT_FLOAT64_MAXERROR ",delta);all(isfinite,got)&&delta<=.125||error("original dequantization numerical criterion");got
 end
 println("STUDY_CONTRACT active_quantized_mode=",packed ? "int4" : "int8"," n512 block32x4 FP16vector_scale_zp_output_FP32accum full512outputs exactlane_emulator originalabsolute_tol0.125 rawFP16disabled_inmain")
 OwnedStudy.run_trials!(operation!,()->fill!(res,Float16(0)),readback,expected,10;atol=0.,rtol=0.)
end

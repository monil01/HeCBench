using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function conv_study(factory)
 w=512;h=256;n=w*h;repeats=10
 input=Float32[Float32(mod(17*i+13*div(i,w),129)-64)/64f0 for i in 0:n-1]
 coeff=Float32[Float32(mod(7*i,17)-8)/32f0 for i in 0:16]
 row=zeros(Float32,n);expected=zeros(Float32,n)
 for y in 0:h-1,x in 0:w-1
  sum=0.;for k in -8:8;if 0<=x+k<w;sum+=Float64(input[y*w+x+k+1])*Float64(coeff[9-k]);end;end
  row[y*w+x+1]=Float32(sum)
 end
 for y in 0:h-1,x in 0:w-1
  sum=0.;for k in -8:8;if 0<=y+k<h;sum+=Float64(row[(y+k)*w+x+1])*Float64(coeff[9-k]);end;end
  expected[y*w+x+1]=Float32(sum)
 end
 di=CuArray(input);dk=CuArray(coeff);db=CUDA.zeros(Float32,n);out=CUDA.zeros(Float32,n);launch! = factory(out,db,di,dk,w,h)
 OwnedStudy.run_trials!(launch!,()->(fill!(out,0f0);fill!(db,0f0)),()->vcat(Array(db),Array(out)),vcat(row,expected),repeats;atol=1e-5,rtol=0.)
 got=Array(out);l2=sqrt(sum((Float64.(got).-Float64.(expected)).^2)/sum(Float64.(expected).^2));println("STUDY_ACCURACY native_relative_L2=",l2," required_below=1e-6 pass=",l2<1e-6);l2<1e-6||error("originalL2criterion failed")
end

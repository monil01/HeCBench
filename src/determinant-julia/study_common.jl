using CUDA,LinearAlgebra
const STUDY_REGION="h2d_factorization_device_diag_product_scalar_d2h_per_repeat"
const SOURCE_EXPECTED_DETERMINANT=2048f0
function matrix_a()
    return Float32[
         2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2;
        -2  4  0  0  0  0  0  0  0  0  0;
        -2  0  6  2  2  2  2  2  2  2  2;
        -2  0  2  8  4  4  4  4  4  4  4;
        -2  0  2  4 10 10 10 10 10 10 10;
        -2  0  2  4 10 12 12 12 12 12 12;
        -2  0  2  4 10 12 14 14 14 14 14;
        -2  0  2  4 10 12 14 16 16 16 16;
        -2  0  2  4 10 12 14 16 18 18 18;
        -2  0  2  4 10 12 14 16 18 20 20;
        -2  0  2  4 10 12 14 16 18 20 22
    ]
end

include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"),String),"kernel_launch_loop_per_internal_repeat"=>STUDY_REGION),joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function determinant_study(factory)
 repeats=10;host=matrix_a();n=11
 oracle=Matrix(cholesky(Hermitian(Float64.(host),:L)).L);expected=Float32[oracle[row,col] for col in 1:n for row in col:n]
 append!(expected,Float32[prod(diag(oracle)),prod(diag(oracle))^2])
 a=CuArray(host);product=CUDA.zeros(Float32,1);scalar=zeros(Float32,1);info=CUDA.zeros(Cint,1)
 solver=CUDA.cuSOLVER.cusolverDnCreate();CUDA.cuSOLVER.cusolverDnSetStream(solver,CUDA.stream());size=Ref{Cint}(0)
 CUDA.cuSOLVER.cusolverDnSpotrf_bufferSize(solver,'L',n,a,n,size);workspace=CuArray{Float32}(undef,Int(size[]))
 reduce! = factory(a,product)
 launch! = ()->begin
  copyto!(a,host)
  CUDA.cuSOLVER.cusolverDnSpotrf(solver,'L',n,a,n,workspace,size[],info)
  reduce!()
  copyto!(scalar,product)
 end
 readback=()->begin
  Array(info)[1]==0||error("cuSOLVER info failure")
  h=Array(a);got=Float32[h[row,col] for col in 1:n for row in col:n]
  maximum(abs.(Float64.(got).-Float64.(expected[1:66])))<=1e-5||error("factor strictCPU check")
  root=scalar[1];det=root*root
  isfinite(det)&&abs(det-SOURCE_EXPECTED_DETERMINANT)<0.05f0||error("originaldetcriterion")
  append!(got,Float32[root,det]);got
 end
 try
  OwnedStudy.run_trials!(launch!,()->nothing,readback,expected,repeats;atol=.05,rtol=0.)
 finally
  CUDA.cuSOLVER.cusolverDnDestroy(solver)
 end
end

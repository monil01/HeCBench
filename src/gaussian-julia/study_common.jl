using CUDA,LinearAlgebra
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function backsolve(a,b,n)
 x=zeros(eltype(a),n)
 for row in n-1:-1:0
  value=b[row+1]
  for col in row+1:n-1;value-=a[row*n+col+1]*x[col+1];end
  x[row+1]=value/a[row*n+row+1]
 end
 x
end
function gaussian_cpu(host,n)
 a=copy(host);b=ones(Float32,n);m=zeros(Float32,n*n)
 for t in 0:n-2
  for row in t+1:n-1;m[row*n+t+1]=a[row*n+t+1]/a[t*n+t+1];end
  for row in t+1:n-1
   for col in t:n-1;a[row*n+col+1]-=m[row*n+t+1]*a[t*n+col+1];end
   b[row+1]-=m[row*n+t+1]*b[t+1]
  end
 end
 vcat(a,b,m,backsolve(a,b,n))
end
function gaussian_study(factory)
 n=128;host=reinterpret(Float32,read(joinpath(@__DIR__,"study_input_f32.bin")))|>collect
 length(host)==n*n||error("fixture length")
 expected=gaussian_cpu(host,n);A=permutedims(reshape(Float64.(host),n,n));exact=A\ones(n)
 maximum(abs.(Float64.(expected[end-n+1:end])-exact))<=1e-3||error("CPU source solution disagrees with independent pivoted Float64 solve")
 a=CuArray(host);b=CUDA.ones(Float32,n);m=CUDA.zeros(Float32,n*n);operation! = factory(a,b,m,n)
 reset! =()->begin;copyto!(a,host);fill!(b,1.f0);fill!(m,0.f0);end
 readback=()->begin
  aa=Array(a);bb=Array(b);mm=Array(m);x=backsolve(aa,bb,n)
  intermediate=vcat(aa,bb,mm);maximum(abs.(Float64.(intermediate)-Float64.(expected[1:end-n])))<=1e-4||error("full intermediate check")
  residual=maximum(abs.(A*Float64.(x).-1))/max(opnorm(A,Inf)*norm(Float64.(x),Inf)+1,1.)
  solution_error=maximum(abs.(Float64.(x)-exact));println("STUDY_SOLVE_CHECK scaled_residual=",residual," solution_abs_error=",solution_error)
  all(isfinite,x)&&isfinite(residual)&&residual<1e-5&&solution_error<=1e-3||error("independent solve/residual check")
  vcat(intermediate,x)
 end
 println("STUDY_CONTRACT n128 nonpivotedGaussianForward254kernels fullA_b_m_and_host_backsolve n33024outputs original_solution_atol1e-3 fullintermediate_atol1e-4 independent_Float64_pivoted_solve_and_residual")
 OwnedStudy.run_trials!(operation!,reset!,readback,expected,1;atol=1e-3,rtol=0.)
end

using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function jacobian_study(factory)
 mode=isempty(ARGS) ? "k1" : ARGS[1];dim=length(ARGS)<2 ? 128 : parse(Int,ARGS[2]);batch=8
 host=Float32[Float32(mod(i*37+11,193)-96)/32.f0 for i in 0:batch*dim-1]
 expected=zeros(Float64,batch*dim*dim);finite_diff_error=0.
 for b in 0:batch-1
  x=Float64.(host[b*dim+1:(b+1)*dim]);ex=exp.(x.-maximum(x));y=ex./sum(ex)
  for i in 0:dim-1,j in 0:dim-1;expected[b*dim*dim+i*dim+j+1]=y[i+1]*((i==j ? 1. : 0.)-y[j+1]);end
  i=mod(b*7+3,dim)+1;j=mod(b*11+5,dim)+1;h=1e-5
  xp=copy(x);xm=copy(x);xp[j]+=h;xm[j]-=h;ep=exp.(xp.-maximum(xp));em=exp.(xm.-maximum(xm));fd=(ep[i]/sum(ep)-em[i]/sum(em))/(2h)
  finite_diff_error=max(finite_diff_error,abs(fd-expected[b*dim*dim+(i-1)*dim+j]))
 end
 finite_diff_error<1e-9||error("independentfinite differences");input=CuArray(host);out=CUDA.zeros(Float32,length(expected));operation=factory(input,out,dim,batch,mode)
 readback=()->begin
  got=Array(out);rowerr=0.;symmetry=0.;minquadratic=Inf
  for b in 0:batch-1
   quad=0.
   for i in 0:dim-1
    rowsum=0.;vi=Float64(mod(i,7)-3)
    for j in 0:dim-1
     value=got[b*dim*dim+i*dim+j+1];isfinite(value)||error("nonfiniteJacobian");rowsum+=Float64(value);symmetry=max(symmetry,abs(Float64(value)-got[b*dim*dim+j*dim+i+1]));quad+=vi*Float64(value)*Float64(mod(j,7)-3)
     (i==j ? value>0 : value<0)||error("Jacobian signs")
    end
    rowerr=max(rowerr,abs(rowsum))
   end
   minquadratic=min(minquadratic,quad)
  end
  rowerr<1e-6&&symmetry<1e-6&&minquadratic>=-1e-6||error("Jacobian invariant")
  println("STUDY_INVARIANTS max_rowsum=",rowerr," max_symmetry=",symmetry," minimum_probe_quadratic_form=",minquadratic," independent_finite_difference_max_abs=",finite_diff_error)
  got
 end
 OwnedStudy.run_trials!(operation,()->fill!(out,0.f0),readback,expected,10;atol=1e-6,rtol=0.)
end

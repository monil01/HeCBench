using CUDA,Printf,SparseArrays,LinearAlgebra
const STUDY_NUM=256
function corrected_coeffs(n)
 a=[zeros(Float32,n*n) for _=1:6];aP,aW,aE,aS,aN,b=a;g=.01f0
 for col=0:n-1,row=0:n-1
  k=col*n+row+1;sp=0f0
  if col==0;sp-=2f0*g;else;aW[k]=g;end
  if col==n-1;sp-=2f0*g;else;aE[k]=g;end
  if row==0;sp-=2f0*g;else;aS[k]=g;end
  if row==n-1;sp-=2f0*g;b[k]=2f0*g;else;aN[k]=g;end
  aP[k]=aW[k]+aE[k]+aS[k]+aN[k]-sp
 end
 return a
end
function cpu_sor(a)
 n=STUDY_NUM;h=n÷2+2;s=h*(n+2);r=zeros(Float32,s);b=similar(r);fill!(b,0);omega=1.85f0
 for it=1:1000000
  sums=Float32[]
  for color=1:2
   dest=color==1 ? r : b;src=color==1 ? b : r;total=0f0
   for col=1:n,row=1:n÷2
    parity=color==1 ? col&1 : (col+1)&1;opposite=1-parity;idx=col*h+row+1;k=2*row-parity+n*(col-1)
    old=dest[idx];res=a[6][k]+a[2][k]*src[row+(col-1)*h+1]+a[3][k]*src[row+(col+1)*h+1]+a[4][k]*src[row-parity+col*h+1]+a[5][k]*src[row+opposite+col*h+1]
    v=old*(1f0-omega)+omega*(res/a[1][k]);dest[idx]=v;diff=v-old;total+=diff*diff
   end
   push!(sums,total)
  end
  rms=sqrt((sums[1]+sums[2])/Float32(n*n))
  if rms<1f-6;@printf("LAPLACE_CPU iterations=%d change_rms=%.12g\n",it,rms);return vcat(r,b);end
 end
 error("CPU convergence")
end
function independent_system(a)
 n=STUDY_NUM;I=Int[];J=Int[];V=Float64[]
 for col=1:n,row=1:n
  k=(col-1)*n+row;push!(I,k);push!(J,k);push!(V,Float64(a[1][k]))
  for (slot,neighbor) in ((2,k-n),(3,k+n),(4,k-1),(5,k+1))
   if a[slot][k]!=0;push!(I,k);push!(J,neighbor);push!(V,-Float64(a[slot][k]));end
  end
 end
 A=sparse(I,J,V,n*n,n*n);issymmetric(A)||error("FVM symmetry");rhs=Float64.(a[6]);exact=cholesky(Symmetric(A))\rhs
 return A,rhs,exact
end
function physical_field(vals)
 n=STUDY_NUM;h=n÷2+2;s=h*(n+2);out=zeros(Float64,n*n)
 for col=1:n,row=1:n
  red=iseven(row+col);packed=red ? (row+(col&1))÷2 : (row+((col+1)&1))÷2
  out[(col-1)*n+row]=Float64(vals[(red ? 0 : s)+col*h+packed+1])
 end
 return out
end
function laplace_study(launch!)
 n=STUDY_NUM;S=(n÷2+2)*(n+2);a=corrected_coeffs(n);prefix=vcat(a...);expected=cpu_sor(a);A,rhs,exact=independent_system(a)
 da=map(CuArray,a);r=CUDA.zeros(Float32,S);b=CUDA.zeros(Float32,S);norm=CUDA.zeros(Float32,S)
 launch!(1,da,r,b,norm);CUDA.sum(norm);launch!(2,da,r,b,norm);CUDA.sum(norm);CUDA.synchronize();allok=true
 for trial=1:5
  fill!(r,0);fill!(b,0);fill!(norm,0);CUDA.synchronize();start=time_ns();iterations=0;rms=0f0
  for it=1:1000000
   iterations=it;launch!(1,da,r,b,norm);nr=CUDA.sum(norm);launch!(2,da,r,b,norm);nb=CUDA.sum(norm);rms=sqrt((nr+nb)/Float32(n*n))
   if rms<1f-6;break;end
  end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9;vals=vcat(Array(r),Array(b));err=maximum(abs.(vals.-expected));field=physical_field(vals);science=maximum(abs.(field.-exact));residual=maximum(abs.(A*field-rhs));ok=iterations<=1000000&&rms<1f-6&&all(isfinite,vals)&&err<=1f-4&&minimum(vals)>=0&&maximum(vals)<=1.00001f0&&science<=2e-3
  allok&=ok;@printf("LAPLACE_SCIENCE trial=%d solver_error_inf=%.12g equation_residual_inf=%.12g min=%.12g max=%.12g\n",trial,science,residual,minimum(field),maximum(field))
  out=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"iterations\":%d,\"iterate_change_rms\":%.12g,\"independent_solver_error_inf\":%.12g,\"equation_residual_inf\":%.12g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",length(vals),err,iterations,rms,science,residual,path,length(out))
 end
 allok||error("Laplace signed/fullstate/independent physical mismatch");println("PASS")
end

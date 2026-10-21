using CUDA,Printf,Libdl
const BICG_LIB=abspath(ARGS[1])
checkstatus(x)=x==0||error("library call status $x")
bcopy(x,y)=checkstatus(ccall((:study_copy,BICG_LIB),Cint,(Cint,Cint),x,y))
bzero(x)=checkstatus(ccall((:study_zero,BICG_LIB),Cint,(Cint,),x))
bdot(x,y)=ccall((:study_dot,BICG_LIB),Cdouble,(Cint,Cint),x,y)
bnorm(x)=ccall((:study_norm,BICG_LIB),Cdouble,(Cint,),x)
baxpy(x,y,a)=checkstatus(ccall((:study_axpy,BICG_LIB),Cint,(Cint,Cint,Cdouble),x,y,a))
bscale(x,a)=checkstatus(ccall((:study_scale,BICG_LIB),Cint,(Cint,Cdouble),x,a))
bmv(x,y,a,b)=checkstatus(ccall((:study_mv,BICG_LIB),Cint,(Cint,Cint,Cdouble,Cdouble),x,y,a,b))
bsolve(upper,x,y)=checkstatus(ccall((:study_solve,BICG_LIB),Cint,(Cint,Cint,Cint),upper,x,y))
# Literal source recurrence, including resetting P=R every iteration.
# Individual CABI calls contain only library primitives, not this solver loop.
function translated_solver()
 checkstatus(ccall((:study_prepare,BICG_LIB),Cint,()))
 bcopy(0,2);bmv(1,2,-1.,1.);delta=bdot(2,2);delta_prev=delta;bcopy(2,3)
 nrm_R=bnorm(2);threshold=1e-10*nrm_R;alpha=0.;omega=0.;iterations=0
 for i in 1:100
  iterations=i;bcopy(3,4)
  if i>1
   delta=bdot(2,3);beta=(delta/delta_prev)*(alpha/omega);delta_prev=delta
   baxpy(8,4,-omega);bscale(4,beta);baxpy(3,4,1.)
  end
  bzero(10);bzero(5);bsolve(0,4,10);bsolve(1,10,5)
  bmv(5,8,1.,0.);denominator=bdot(2,8);alpha=delta/denominator
  baxpy(5,1,alpha);bcopy(3,6);baxpy(8,6,-alpha);nrm_S=bnorm(6)
  if nrm_S<threshold;break;end
  bzero(10);bzero(7);bsolve(0,6,10);bsolve(1,10,7);bmv(7,9,1.,0.)
  omega_num=bdot(9,6);omega_den=bdot(9,9);omega=omega_num/omega_den
  baxpy(7,1,omega);bcopy(6,3);baxpy(9,3,-omega);nrm_R=bnorm(3)
  if nrm_R<threshold;break;end
 end
 bcopy(0,3);bmv(1,3,-1.,1.);nrm_R=bnorm(3)
 checkstatus(ccall((:study_finish,BICG_LIB),Cint,()))
 (nrm_R,iterations)
end
function bicg_study()
 checkstatus(ccall((:study_init,BICG_LIB),Cint,(Cstring,),abspath(ARGS[2])))
 for k in 0:1
  p=unsafe_string(ccall((:study_library,BICG_LIB),Cstring,(Cint,),k));v=ccall((:study_version,BICG_LIB),Cint,(Cint,),k)
  println("STUDY_LIBRARY kind=",k," path=",p," version=",v);flush(stdout)
 end
 CUDA.device_synchronize(blocking=true)
 n=ccall((:study_capture_count,BICG_LIB),Cint,());before=zeros(Float64,n);after=similar(before);metrics=zeros(3)
 # Warm the exact complete Julia orchestration and all first-use ccall/JIT paths.
 checkstatus(ccall((:study_reset,BICG_LIB),Cint,()));translated_solver();CUDA.device_synchronize(blocking=true)
 for trial in 1:5
  checkstatus(ccall((:study_reset,BICG_LIB),Cint,()));checkstatus(ccall((:study_capture,BICG_LIB),Cint,(Ptr{Cdouble},),before))
  CUDA.device_synchronize(blocking=true);t=time_ns();norm,iterations=translated_solver();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-t)*1e-9
  checkstatus(ccall((:study_capture,BICG_LIB),Cint,(Ptr{Cdouble},),after))
  ok=ccall((:study_check,BICG_LIB),Cint,(Ptr{Cdouble},Ptr{Cdouble},Cdouble,Cint,Ptr{Cdouble}),before,after,norm,iterations,metrics)==1
  output=vcat(before,after,Float64[iterations,norm]);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"bicgstab_trial$(trial).bin");write(path,output)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"independent_true_residual\":%.17g,\"iterations\":%d,\"jit_excluded\":true,\"timing_region\":\"full_BiCGStab_SpSV_analysis_allocation_host_control_iterations_final_residual_cleanup\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(output),metrics[1],metrics[2],metrics[3],iterations,path,length(output));flush(stdout);ok||error("independent full stencil/ILU/solution/guards checks")
 end
 checkstatus(ccall((:study_destroy,BICG_LIB),Cint,()))
end

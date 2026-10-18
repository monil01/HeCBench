using CUDA,Printf
const ROOT=@__DIR__
const CASES=(Geometry(8,8,1,8,8,1,1,1,1,1,1,8,0,-.5),Geometry(7,8,2,8,8,3,.9,1,.5,1,1,20,2,1))
const ANGLES=(([0.],[0.]),([-5.,0.,4.],[-2.,.5,1.5]))
const INPUTS=[collect(reinterpret(Float64,read(joinpath(ROOT,"study_proj$(i)_f64.bin")))) for i in 1:2]
const ORACLES=[collect(reinterpret(Float64,read(joinpath(ROOT,"oracle_case$(i)_f64.bin")))) for i in 1:2]
function make_arrays(p)
 X=p.ndx+1;Y=p.ndy+1;OX=p.nx+1;OY=p.ny+1;sizes=(X*Y*p.np,OX*OY,p.nx*p.ny*p.ns,X,Y,Y,OX,OY,p.ns,Y,X*Y,Y,Y)
 bases=map(n->CUDA.fill(-8192.,n+32),sizes);arrays=map((b,n)->view(b,17:16+n),bases,sizes)
 arrays,bases
end
function pipeline!(launch,A,p,data,angles,capture)
 X=p.ndx+1;Y=p.ndy+1;OX=p.nx+1;OY=p.ny+1
 for np in 0:p.np-1
  launch(Val(1),(div(X,256)+1,1,1),(256,1,1),A,p,Aux(np=np))
  fill!(view(A[1],np*X*Y+2:np*X*Y+1+p.ny),0.)
 end
 for np in 0:p.np-1,c in 0:p.ndx-1
  copyto!(A[1],np*X*Y+(c+1)*Y+2,data,np*p.ndx*p.ndy+c*p.ndy+1,p.ndy)
 end
 capture!==nothing&&capture(:input,A[1])
 for (slot,n,left,step,offset) in ((4,X,Float64(p.ndx),-p.du,0.),(5,Y,p.ndy/2,p.dv,0.),(7,OX,Float64(p.nx),-p.dx,0.),(8,OY,p.ny/2,p.dy,0.),(9,p.ns,0.,p.dz,p.DAG+p.dz/2))
  launch(Val(2),(div(n-1,256)+1,1,1),(256,1,1),A,p,Aux(slot=slot,count=n,left=left,step=step,offset=offset))
 end
 fill!(A[6],0.);fill!(A[3],0.);capture!==nothing&&capture(:state,A[4],A[5],A[6],A[7],A[8],A[9])
 launch(Val(5),(cld(Y*p.np,256),1,1),(256,1,1),A,p,Aux(slot=1));capture!==nothing&&capture(:state,A[1])
 launch(Val(5),(cld(X*p.np,256),1,1),(256,1,1),A,p,Aux(slot=0));capture!==nothing&&capture(:state,A[1])
 for np in 0:p.np-1
  theta=angles[1][np+1]*pi/180;phi=angles[2][np+1]*pi/180;ty=-(p.DSD-p.DDR)*sin(theta);tz=(p.DSD-p.DDR)*cos(theta)+p.DDR
  launch(Val(3),(div(Y,256)+1,1,1),(256,1,1),A,p,Aux(np=np,phi=phi));capture!==nothing&&capture(:state,A[12],A[13])
  for nz in 0:p.ns-1
   v=Aux(np=np,nz=nz,ty=ty,tz=tz)
   launch(Val(4),(div(Y,16)+1,div(X,16)+1,1),(16,16,1),A,p,v);capture!==nothing&&capture(:state,A[11],A[10])
   launch(Val(6),(div(OY,16)+1,div(OX,16)+1,1),(16,16,1),A,p,v);capture!==nothing&&capture(:state,A[2])
   launch(Val(7),(div(p.ny,16)+1,div(p.nx,16)+1,1),(16,16,1),A,p,v);capture!==nothing&&capture(:state,A[3])
  end
 end
 launch(Val(8),(div(p.ny,8)+1,div(p.nx,8)+1,div(p.ns,4)+1),(8,8,4),A,p,Aux());capture!==nothing&&capture(:state,A[3])
 nothing
end
function ddbp_study(launch)
 prepared=map(make_arrays,CASES);allok=true
 for ci in 1:2;pipeline!(launch,prepared[ci][1],CASES[ci],INPUTS[ci],ANGLES[ci],nothing);end
 CUDA.device_synchronize(blocking=true)
 for trial in 1:5
  prefix=Float64[];states=Float64[];guards=Float64[];times=zeros(2);err=0.;rel=0.;ok=true
  for ci in 1:2
   p=CASES[ci];A,bases=prepared[ci];CUDA.device_synchronize(blocking=true);start=time_ns()
   for _ in 1:3;pipeline!(launch,A,p,INPUTS[ci],ANGLES[ci],nothing);end
   CUDA.device_synchronize(blocking=true);times[ci]=(time_ns()-start)*1e-9/3
   localstates=Float64[];localprefix=Float64[]
   capture=function(kind,arrays...)
    CUDA.device_synchronize(blocking=true)
    for a in arrays;append!(kind==:input ? localprefix : localstates,Array(a));end
   end
   pipeline!(launch,A,p,INPUTS[ci],ANGLES[ci],capture);CUDA.device_synchronize(blocking=true)
   append!(localprefix,ANGLES[ci][1],ANGLES[ci][2],Float64[getfield(p,f) for f in fieldnames(Geometry)])
   append!(prefix,localprefix);append!(states,localstates);oracle=ORACLES[ci];@assert length(localstates)==length(oracle)
   errors=abs.(localstates.-oracle);err=max(err,maximum(errors));rel=max(rel,maximum(errors./max.(abs.(oracle),eps(Float64))));ok&=all(isfinite,localstates)&&all(errors.<=1e-10.+1e-9.*abs.(oracle))
   for base in bases;raw=Array(base);g=vcat(raw[1:16],raw[end-15:end]);append!(guards,g);ok&=all(==(-8192.),g);end
  end
  allok&=ok;artifact=vcat(prefix,states,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":3699,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_two_corrected_backprojection_original_preallocated_padding_H2D_maps_scans_geometry_interpolation_differentiation_normalization_regions\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_cases\":2,\"component_time_seconds\":{\"analytic_plane\":%.12g,\"varied_views\":%.12g},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",err,rel,times[1],times[2],path,length(artifact))
 end
 allok||error("Fullbackprojectionstage/oracle/guard failure");println("PASS")
end

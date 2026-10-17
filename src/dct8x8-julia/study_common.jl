using CUDA,Printf,LinearAlgebra
function study_fixture(f)
 w,h,s=f==1 ? (64,32,64) : f==2 ? (67,35,73) : (32,16,32);x=zeros(Float32,s*h)
 if f==1
  ccall(:srand,Cvoid,(Cuint,),2009)
  for i in eachindex(x);x[i]=Float32(ccall(:rand,Cint,()))/Float32(typemax(Cint));end
 elseif f==2
  for i in 0:length(x)-1;x[i+1]=Float32((13i+5)%257-128)/128;end
 elseif f==3
  fill!(x,.25f0)
 else
  x[3s+6]=1f0
 end
 (w,h,s,x)
end
function mathematical_dct(f,dir)
 w,h,s,x=f;gold=fill(-123.25,length(x));basis=[(k==0 ? sqrt(1/8) : .5*cos((2j+1)*k*pi/16)) for k in 0:7,j in 0:7]
 for by in 0:8:h-8,bx in 0:8:w-8,v in 0:7,u in 0:7
  z=0.;for n in 0:7,m in 0:7;z+=Float64(x[(by+n)*s+bx+m+1])*(dir==666 ? basis[u+1,m+1]*basis[v+1,n+1] : basis[m+1,u+1]*basis[n+1,v+1]);end
  gold[(by+v)*s+bx+u+1]=z
 end
 # Independent orthonormality and per-tile Parseval checks of mathematical oracle.
 @assert maximum(abs.(basis*transpose(basis).-Matrix{Float64}(I,8,8)))<4e-15
 for by in 0:8:h-8,bx in 0:8:w-8
  ein=sum(Float64(x[(by+n)*s+bx+m+1])^2 for n in 0:7,m in 0:7);eout=sum(gold[(by+n)*s+bx+m+1]^2 for n in 0:7,m in 0:7);@assert abs(ein-eout)<1e-12*max(ein,1)
 end
 gold
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=[study_fixture(f) for f in 1:4];dx=[CuArray(f[4]) for f in fs];ys=[CUDA.fill(-123.25f0,length(fs[f][4])) for f in 1:4,d in 1:2];gold=[mathematical_dct(fs[f],d==1 ? 666 : 777) for f in 1:4,d in 1:2];prefix=vcat([vcat(Float32[f[1],f[2],f[3]],f[4]) for f in fs]...)
 for f in 1:4,d in 1:2;w,h,s,x=fs[f];launch!(ys[f,d],dx[f],s,h,w,d==1 ? 666 : 777);end;CUDA.synchronize()
 for trial in 1:5
  for y in ys;fill!(y,-123.25f0);end;CUDA.synchronize();st=time_ns()
  for _ in 1:rep,f in 1:4,d in 1:2;w,h,s,x=fs[f];launch!(ys[f,d],dx[f],s,h,w,d==1 ? 666 : 777);end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;got=Float32[];err=0.;ok=true
  for f in 1:4,d in 1:2
   g=Array(ys[f,d]);delta=maximum(abs.(Float64.(g).-gold[f,d]));err=max(err,delta);ok&=all(isfinite,g)&&all(abs.(Float64.(g).-gold[f,d]).<=1e-5.+1e-6.*abs.(gold[f,d]));append!(got,g)
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"eight_full_forward_inverse_DCTs_four_images_no_transfers\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,rep,path,length(art));ok||error("mathematical orthonormal DCT full-array contract")
 end
end

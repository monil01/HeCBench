const STUDY_ITERS=8
function fixture(perturbed)
 n=WIDTH*HEIGHT;a=zeros(Float64,n);b=zeros(Float64,4n);c=zeros(Float64,4n);types=zeros(UInt8,n)
 for y in 0:HEIGHT-1,x in 0:WIDTH-1
  pos=x+WIDTH*y+1;rho=perturbed ? 1+mod(17x+13y,9)/64 : 1.;ux=perturbed ? 1/128 : 0.;uy=perturbed ? -1/256 : 0.
  types[pos]=(x==0||x==WIDTH-1||y==0||y==HEIGHT-1||(perturbed&&x in 120:123&&y in 12:19)) ? 1 : 0
  a[pos]=ced(rho,WEIGHT[1],0,0,ux,uy)
  for k in 1:4;b[(pos-1)*4+k]=ced(rho,WEIGHT[k+1],DIRX[k+1],DIRY[k+1],ux,uy);c[(pos-1)*4+k]=ced(rho,WEIGHT[k+5],DIRX[k+5],DIRY[k+5],ux,uy);end
 end
 a,b,c,types
end
function encode(f)
 vcat([f[1,p] for p in 1:WIDTH*HEIGHT],[f[k,p] for p in 1:WIDTH*HEIGHT for k in 2:5],[f[k,p] for p in 1:WIDTH*HEIGHT for k in 6:9])
end
function expected(a,b,c,types)
 n=WIDTH*HEIGHT;f=zeros(Float64,9,n)
 for p in 1:n
  f[1,p]=a[p]
  for k in 1:4;f[k+1,p]=b[(p-1)*4+k];f[k+5,p]=c[(p-1)*4+k];end
 end
 out=copy(f);opposite=(1,4,5,2,3,8,9,6,7)
 for _ in 1:STUDY_ITERS
  collision=similar(f)
  for pos in 1:n
   if types[pos]==1
    for k in 1:9;collision[k,pos]=f[opposite[k],pos];end
   else
    rho=sum(@view f[:,pos]);ux=sum(f[k,pos]*DIRX[k] for k in 1:9)/rho;uy=sum(f[k,pos]*DIRY[k] for k in 1:9)/rho
    for k in 1:9;collision[k,pos]=(1-OMEGA)*f[k,pos]+OMEGA*ced(rho,WEIGHT[k],DIRX[k],DIRY[k],ux,uy);end
   end
  end
  # Pull each population from its source; skip outer-edge source exactly as native guard.
  for y in 0:HEIGHT-1,x in 0:WIDTH-1,k in 1:9
   sx=x-DIRX[k];sy=y-DIRY[k]
   if 0<sx<WIDTH-1&&0<sy<HEIGHT-1;out[k,x+WIDTH*y+1]=collision[k,sx+WIDTH*sy+1];end
  end
  f,out=out,f
 end
 vcat(encode(f),encode(out))
end
function study(run!)
 cases=[];prefix=Float64[];w=CuArray(collect(WEIGHT))
 for perturbed in(false,true)
  a,b,c,types=fixture(perturbed);append!(prefix,a);append!(prefix,b);append!(prefix,c);append!(prefix,Float64.(types));append!(prefix,[OMEGA,WEIGHT...,Float64(WIDTH),Float64(HEIGHT)])
  push!(cases,(a,b,c,CuArray(types),expected(a,b,c,types),CuArray(a),CuArray(b),CuArray(c),CuArray(a),CuArray(b),CuArray(c)))
 end
 function case!(args)
  a,b,c,types,want,da,db,dc,oa,ob,oc=args
  for(dst,src)in((da,a),(db,b),(dc,c),(oa,a),(ob,b),(oc,c));copyto!(dst,src);end
  CUDA.synchronize();t=time_ns()
  for _ in 1:STUDY_ITERS
   run!(da,oa,db,ob,dc,oc,types,w,OMEGA)
   da,oa=oa,da;db,ob=ob,db;dc,oc=oc,dc
  end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/STUDY_ITERS;got=vcat(Array(da),Array(db),Array(dc),Array(oa),Array(ob),Array(oc));err=maximum(abs.(got.-want));return got,elapsed,err
 end
 for args in cases;case!(args);end
 for trial in 1:5
  output=copy(prefix);elapsed=0.;err=0.
  for args in cases;got,t,e=case!(args);append!(output,got);elapsed+=t;err=max(err,e);end
  ok=all(isfinite,output)&&err<=1e-10;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":458776,\"max_abs_error\":",err,",\"timing_region\":\"sum_two_fixtures_eightstep_lbm_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":8,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":458776,\"byte_order\":\"little\"}}")
  ok||Base.error("FullD2Q9populations/scratchoraclefailed")
 end
end

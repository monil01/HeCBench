using CUDA,Printf
if !isdefined(@__MODULE__,:BLOSUM62);include(joinpath(@__DIR__,"study_blosum.jl"));end
function study_fixture(f)
 dim,p=f==1 ? (64,10) : f==2 ? (96,3) : (32,2);cols=dim+1
 if f==1
  ccall(:srand,Cvoid,(Cuint,),7);s1=Int32[ccall(:rand,Cint,())%10+1 for _ in 1:dim];s2=Int32[ccall(:rand,Cint,())%10+1 for _ in 1:dim]
 elseif f==2
  s1=Int32[(7i+3)%24 for i in 0:dim-1];s2=Int32[(5i+11)%24 for i in 0:dim-1]
 else
  s1=zeros(Int32,dim);s2=copy(s1)
 end
 ref=zeros(Int32,cols^2);initial=zeros(Int32,cols^2)
 for i in 1:dim;initial[i*cols+1]=-i*p;initial[i+1]=-i*p;end
 for i in 1:dim,j in 1:dim;ref[i*cols+j+1]=BLOSUM62[s1[i]+1,s2[j]+1];end
 # Independent Int64 rowwise DP, unlike original tiled native reference.
 gold=Int64.(initial)
 for i in 1:dim,j in 1:dim
  k=i*cols+j+1;gold[k]=max(gold[k-cols-1]+ref[k],gold[k-1]-p,gold[k-cols]-p)
 end
 if f==3
  @assert all(gold[i*cols+j+1]==4min(i,j)-p*abs(i-j) for i in 0:dim,j in 0:dim)
 end
 (dim,p,s1,s2,ref,initial,gold)
end
function traceback_check(got,fixture)
 dim,p,s1,s2,ref,initial,gold=fixture;cols=dim+1;i=dim;j=dim;score=0;consumed1=0;consumed2=0
 while i>0||j>0
  v=got[i*cols+j+1]
  if i>0&&j>0&&v==got[(i-1)*cols+j]+BLOSUM62[s1[i]+1,s2[j]+1]
   score+=BLOSUM62[s1[i]+1,s2[j]+1];i-=1;j-=1;consumed1+=1;consumed2+=1
  elseif j>0&&v==got[i*cols+j]-p
   score-=p;j-=1;consumed2+=1
  elseif i>0&&v==got[(i-1)*cols+j+1]-p
   score-=p;i-=1;consumed1+=1
  else
   return false
  end
 end
 score==got[end]&&consumed1==dim&&consumed2==dim
end
function study_main(launch!)
 repeat=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fixtures=[study_fixture(f) for f in 1:3];ds=[CuArray(f[6]) for f in fixtures];dr=[CuArray(f[5]) for f in fixtures];prefix=vcat([vcat(f[5],f[6],f[3],f[4],Int32[f[1],f[2]]) for f in fixtures]...)
 for f in 1:3;launch!(ds[f],dr[f],fixtures[f][1],fixtures[f][2]);end;CUDA.synchronize()
 for trial in 1:5
  for f in 1:3;copyto!(ds[f],fixtures[f][6]);end;CUDA.synchronize();st=time_ns()
  for _ in 1:repeat,f in 1:3;launch!(ds[f],dr[f],fixtures[f][1],fixtures[f][2]);end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/repeat;got=Int32[];ok=true
  for f in 1:3
   g=Array(ds[f]);ok&=g==fixtures[f][7]&&traceback_check(g,fixtures[f]);append!(got,g)
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"three_complete_global_alignment_matrices_no_H2D_inside\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),repeat,path,length(art));ok||error("NeedlemanWunsch fullmatrix/traceback contract")
 end
end

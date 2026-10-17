using CUDA,Printf
function fixture(f)
 m,n,barg=f==1 ? (37,29,3) : (19,17,-1);b=barg<=0 ? 1 : barg;rad=m÷4;seq=zeros(Int32,n);centers=zeros(Int32,n);a=zeros(Int32,m*n*b)
 if f==1
  ccall(:srand,Cvoid,(Cuint,),123)
  for i in eachindex(seq);seq[i]=ccall(:rand,Cint,())%(m÷2);end
  for i in eachindex(centers);centers[i]=ccall(:rand,Cint,())%m;end
  for i in eachindex(a);a[i]=ccall(:rand,Cint,())%(m*n);end
 else
  for j in 0:n-1;seq[j+1]=Int32((0,m,m÷2)[j%3+1]);centers[j+1]=Int32((0,m-1,m÷2)[j%3+1]);end
  for i in 0:length(a)-1;a[i+1]=Int32(2i+17);end
 end
 blocks=barg<=0 ? (n*m)÷256 : n*m
 gold=[similar(a) for _ in 1:6]
 for batch in 0:b-1,row in 0:n-1,col in 0:m-1
  ix=batch*n*m+row*m+col+1
  # Independent shape-indexed mathematical predicates, not flatGPUindexing/helperreuse.
  keeps=(col<seq[row+1],abs(col-centers[row+1])<=rad,col<=row,col>=row,col<row,col>row)
  for mode in 1:6;gold[mode][ix]=keeps[mode] ? a[ix] : Int32(-1);end
 end
 (m,n,barg,b,rad,blocks,seq,centers,a,gold)
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=[fixture(f) for f in 1:2];ds=[(CuArray(f[9]),CuArray(f[7]),CuArray(f[8]),[CuArray{Int32}(undef,length(f[9])) for _ in 1:6]) for f in fs];prefix=vcat([vcat(Int32[f[1],f[2],f[3],f[4],f[5]],f[7],f[8],f[9]) for f in fs]...)
 for (f,d) in zip(fs,ds),mode in 1:6;launch!(d[1],d[2],d[3],d[4][mode],f,Val(mode));end;CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();st=time_ns();for _ in 1:rep,(f,d) in zip(fs,ds),mode in 1:6;launch!(d[1],d[2],d[3],d[4][mode],f,Val(mode));end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;got=Int32[];ok=true
  for (f,d) in zip(fs,ds),mode in 1:6;g=Array(d[4][mode]);ok&=g==f[10][mode];append!(got,g);end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"all6maskoperators_twofixtures_12kernelbatch_sourcegridpolicy\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),rep,path,length(art));ok||error("sixfullintegersequence_window_triangularmaskoutputs")
 end
end

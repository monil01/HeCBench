using CUDA,Libdl
function fixture(m,n)
 dense=Float32[];rows=Int64[0];cols=Int64[];values=Float32[]
 for y in 0:m-1
  for x in 0:n-1
   nz=y%7!=0&&(y%11==1||(y*13+x*7)%9<2);v=nz ? Float32((y*5+x*3)%17+1)/8f0 : ((y%7==0&&x%3==0) ? -0f0 : 0f0)
   if nz&&isodd(x+y);v=-v;end
   push!(dense,v)
   if v!=0;push!(cols,x);push!(values,v);end
  end
  push!(rows,length(cols))
 end
 return dense,rows,cols,values
end
function study(convert,allocate)
 lib=Libdl.dlopen(joinpath(pwd(),"libspd2s_study.so"));prepare=Libdl.dlsym(lib,:study_prepare);destroy=Libdl.dlsym(lib,:study_destroy);version=Libdl.dlsym(lib,:study_sparse_version);println("CuSparseVersion ",ccall(version,Cint,()));cases=[];prefix=Float64[]
 for (m,n) in ((17,19),(65,33))
  dense,rows,cols,values=fixture(m,n);nnz=length(cols);dd=allocate(Float32,length(dense));dr=allocate(Int64,length(rows));dc=allocate(Int64,nnz);dv=allocate(Float32,nnz);copyto!(dd,dense);CUDA.synchronize(CUDA.context());status=Ref{Cint}(-1);ctx=ccall(prepare,Ptr{Cvoid},(CuPtr{Float32},CuPtr{Int64},CuPtr{Int64},CuPtr{Float32},Int64,Int64,Int64,Ref{Cint}),dd,dr,dc,dv,m,n,nnz,status);status[]==0&&ctx!=C_NULL||error("AnalysisFailed");actualrows=Array(dr);actualrows==rows||error("AnalysisOffsetMismatch");append!(prefix,dense);append!(prefix,actualrows);append!(prefix,fill(-1,nnz));append!(prefix,zeros(nnz));append!(prefix,[m,n,nnz,10]);push!(cases,(m,n,rows,cols,values,dd,dr,dc,dv,ctx))
 end
 function case!(c)
  m,n,rows,cols,values,dd,dr,dc,dv,ctx=c;CUDA.fill!(dc,Int64(-1));CUDA.fill!(dv,0f0);CUDA.synchronize(CUDA.context());t=time_ns()
  for _ in 1:10;convert(dd,dr,dc,dv,Int32(m),Int32(n));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9/10;rr=Array(dr);cc=Array(dc);vv=Array(dv)
  for y in 1:m
   first=Int(rr[y])+1;last=Int(rr[y+1]);if first<=last;perm=sortperm(@view cc[first:last]);cc[first:last]=cc[first:last][perm];vv[first:last]=vv[first:last][perm];end
  end
  got=vcat(Float64.(rr),Float64.(cc),Float64.(vv),Float64[length(cc)]);ok=rr==rows&&cc==cols&&vv==values;return got,elapsed,ok
 end
 for c in cases;got,t,ok=case!(c);ok||error("WarmupMismatch");end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=case!(c);append!(out,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum_two_ten_convert_only_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("PairedCSRRowsColumnsValuesMismatch")
 end
 for c in cases;@assert ccall(destroy,Cint,(Ptr{Cvoid},),c[end])==0;end
end

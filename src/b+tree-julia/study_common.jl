using CUDA,Libdl
function study(alloc,point,rangequery)
 cs=open(joinpath(@__DIR__,"fixtures.bin"),"r")do io
  read(io,Int32)==3||error("fixturecasecount");cases=[]
  for ci in 1:3
   n=Int(read(io,Int32));nn=Int(read(io,Int32));h=Int(read(io,Int32));q=Int(read(io,Int32))
   0<=h<=2&&0<n<=65536&&nn>0&&q>0||error("fixtureshape")
   function get(size)
    v=Vector{Int32}(undef,size);read!(io,v);v
   end
   function B(host)
    raw=alloc(fill(eltype(host)(-777),length(host)+128));(host=host,raw=raw,data=view(raw,65:64+length(host)))
   end
   nodes=B(get(nn*517));records=B(get(n));keys=B(get(q));start=B(get(q));finish=B(get(q))
   zero64()=B(zeros(Int64,q));poison32()=B(fill(Int32(-777),q))
   push!(cases,(n=n,nn=nn,h=h,q=q,nodes=nodes,records=records,keys=keys,start=start,finish=finish,pc=zero64(),po=zero64(),ans=poison32(),rc=zero64(),ro=zero64(),last=zero64(),off2=zero64(),rs=poison32(),rl=poison32()))
  end
  eof(io)||error("fixturetrailing");cases
 end
 allbuffers(c)=(c.nodes,c.records,c.keys,c.start,c.finish,c.ans,c.rs,c.rl,c.pc,c.po,c.rc,c.ro,c.last,c.off2)
 initialorder(c)=(c.nodes,c.records,c.keys,c.start,c.finish,c.pc,c.po,c.ans,c.rc,c.ro,c.last,c.off2,c.rs,c.rl)
 outputs(c)=(c.pc,c.po,c.ans,c.rc,c.ro,c.last,c.off2,c.rs,c.rl)
 function reset(c)
  for b in allbuffers(c);fill!(b.raw,eltype(b.host)(-777));copyto!(b.data,b.host);end
 end
 for c in cs;reset(c);point(c);rangequery(c);CUDA.device_synchronize();end
 check=Libdl.dlsym(Libdl.dlopen(joinpath(pwd(),"studycheck.so")),:study_check)
 for trial in 1:5
  out=Int64[3];seconds=0.
  for c in cs
   reset(c);append!(out,Int64[c.n,c.nn,c.h,c.q])
   for b in initialorder(c);append!(out,Array(b.data));end
  end
  for c in cs
   CUDA.device_synchronize();start=time_ns();point(c);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
   start=time_ns();rangequery(c);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
   for b in outputs(c);append!(out,Array(b.data));end
   for b in (c.nodes,c.records,c.keys,c.start,c.finish);Array(b.data)==b.host||error("readonly");end
   for b in allbuffers(c);raw=Array(b.raw);append!(out,raw[1:64]);append!(out,raw[end-63:end]);end
  end
  path="trial$(trial).bin";open(io->write(io,out),path,"w");met=zeros(Float64,2)
  ccall(check,Cint,(Cstring,Ptr{Float64}),path,met)==0||error("independent/fulltreequerycheck")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":",length(out),",\"max_abs_error\":",met[1],",\"max_rel_error\":",met[2],",\"timing_region\":\"sum3_point_and_range_kernel_context_complete\",\"jit_excluded\":true,\"internal_repeats\":6,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
 end
end

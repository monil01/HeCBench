include("reference_history/xs_device_candidate02_exact_Int_conversion.jl")
pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));pos=2
for ci in 1:6
 global pos
 typ,I,G,N,reps,bins,M,U,IX=Int.(pre[pos:pos+8]);pos+=9;num=Int32.(pre[pos:pos+11]);pos+=12;concs=pre[pos:pos+12*M-1];pos+=12*M;mats=Int32.(pre[pos:pos+12*M-1]);pos+=12*M;points=[XPoint(pre[pos+6*i],pre[pos+6*i+1],pre[pos+6*i+2],pre[pos+6*i+3],pre[pos+6*i+4],pre[pos+6*i+5]) for i in 0:I*G-1];pos+=6*I*G;energy=pre[pos:pos+U-1];pos+=U;idx=Int32.(pre[pos:pos+IX-1]);pos+=IX
 if typ==2
  e,mat=xsample(Int32(1));println("candidate02 firstHASHlookup energy=",e," scaledhash=",e/(1.0/bins)," correctCplusplusTruncation=",trunc(Int64,e/(1.0/bins)))
  try
   xprimary!(Int32(1),zeros(Int32,N),num,concs,mats,points,energy,idx,Int64(I),Int64(G),Int32(typ),Int32(bins),Int32(M))
   error("expected conversionwitness absent")
  catch err
   showerror(stdout,err,catch_backtrace());println();err isa InexactError||rethrow()
  end
  break
 end
end

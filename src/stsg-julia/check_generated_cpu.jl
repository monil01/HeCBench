include("filter_device.jl")
const X=Int32(17);const Y=Int32(13);const B=Int32(12);const Z=Int32(3);const W=Int32(81);const N=X*Y
io=open(joinpath(@__DIR__,"cpu_fixture.bin"));raw=read!(io,Vector{Int16}(undef,N*B*Z));rawqa=read!(io,Vector{UInt8}(undef,N*B*Z));@assert eof(io);close(io)
nd=zeros(Float32,N*B*Z);qa=zeros(Float32,N*B*Z);ref=zeros(Float32,N*B);res=zeros(Float32,N*W*4);r3=zeros(Float32,N*W*3);out=zeros(Float32,N*B*Z);vin=zeros(Float32,N*B);res1=zeros(Int32,N*B);index=zeros(Int32,N*W*3)
function eachcell(call)
 for j in 0:Y-1,i in 0:X-1
  call(UInt32(i),UInt32(j),UInt32(1),UInt32(1),UInt32(0),UInt32(0))
 end
end
eachcell((ids...)->Short_to_Float_body!(raw,rawqa,X,Y,B,Z,nd,qa,ids...))
eachcell((ids...)->Generate_NDVI_reference_body!(.8f0,Int32(2),nd,qa,X,Y,B,Z,ref,r3,res1,ids...))
fill!(r3,0f0)
eachcell((ids...)->Compute_d_res_body!(nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,Int32(4),res,ids...))
eachcell((ids...)->STSG_filter_body!(nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,Int32(4),.5f0,Int32(1),out,vin,res,r3,index,ids...))
gold=reinterpret(Float64,read(joinpath(@__DIR__,"cpu_oracle.bin")));got=vcat(Float64.(nd),Float64.(qa),Float64.(ref),Float64.(res),Float64.(r3),Float64.(out),Float64.(vin),Float64.(res1),Float64.(index));@assert length(gold)==length(got)
error=maximum(abs.(got.-gold));bad=count(i->!isfinite(got[i])||abs(got[i]-gold[i])>2e-5+2e-5*abs(gold[i]),eachindex(gold));println("STSG_GENERATED_CPU_CHECK values=",length(got)," max_abs_error=",error," violations=",bad);@assert bad==0

using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function nonzero_study(launch!;args=ARGS)
    nr=256;nc=256;n=nr*nc;r=5;nb=cld(n,256)
    input=Int32[(i*37+11)%5==0 ? 0 : isodd(i) ? -1 : 1 for i in 0:n-1]
    indices=findall(!iszero,input);nz=length(indices);expected=vcat(Int64(nz),Int64.((indices.-1).÷nc),Int64.((indices.-1).%nc))
    din=CuArray(input);out=CUDA.zeros(Int64,2n);prefix=CUDA.zeros(Int32,n);totals=CUDA.zeros(Int32,nb);offsets=similar(totals);count=CUDA.zeros(Int64,1);lastcount=Ref(Int64(0))
    launch()=(lastcount[]=launch!(din,out,prefix,totals,offsets,count,Int32(n),Int32(nb),Int64(nr),Int64(nc));nothing)
    function readback()
        got=Array(out);actual=lastcount[];0<=actual<=n||error("invalid devicecount")
        return vcat(actual,got[1:2actual])
    end
    OwnedStudy.run_trials!(launch,()->nothing,readback,expected,r;atol=0)
end

using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
if !isdefined(Main,:CONFIGS)
    include(joinpath(@__DIR__,"study_types.jl"))
end
function bs_study(factory)
    n=262144;repeats=10
    ty=Int32[];spot=Float32[];strike=Float32[];div=Float32[];risk=Float32[];maturity=Float32[];vol=Float32[];expected=Float32[]
    # IndependentFloat64libm-erf analyticoracle, originalFloat32inputvalues promoted beforeevaluation.
    for i in 0:n-1
        t,K,S,q,r,T,v=CONFIGS[i%37+1];push!(ty,t);push!(strike,K);push!(spot,S);push!(div,q);push!(risk,r);push!(maturity,T);push!(vol,v)
        K64=Float64(K);S64=Float64(S);q64=Float64(q);r64=Float64(r);T64=Float64(T);v64=Float64(v)
        sd=v64*sqrt(T64);d1=(log(S64/K64)+(r64-q64+0.5*v64*v64)*T64)/sd;d2=d1-sd
        N1=0.5*(1+ccall((:erf,Base.Math.libm),Float64,(Float64,),d1/sqrt(2.0)))
        N2=0.5*(1+ccall((:erf,Base.Math.libm),Float64,(Float64,),d2/sqrt(2.0)))
        call=S64*exp(-q64*T64)*N1-K64*exp(-r64*T64)*N2
        price=t==1 ? call-S64*exp(-q64*T64)+K64*exp(-r64*T64) : call
        push!(expected,Float32(price))
    end
    @assert all(>=(0f0),expected)
    output=CUDA.zeros(Float32,n)
    launch! = factory(CuArray(ty),CuArray(spot),CuArray(strike),CuArray(div),CuArray(risk),CuArray(maturity),CuArray(vol),output,Int32(n))
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-4,rtol=0.)
end

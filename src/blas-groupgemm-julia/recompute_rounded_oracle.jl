function main(args=String[])
    groups=4;k=32;n=48;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat")
    ccall(:srand48,Cvoid,(Clong,),123);cdf=cumsum([.2+ccall(:drand48,Cdouble,()) for _ in 1:groups]);m=zeros(Int32,groups)
    for _ in 1:groups*8
        x=ccall(:drand48,Cdouble,())*last(cdf);g=1;while g<groups&&x>cdf[g];g+=1;end;m[g]+=1
    end
    m=max.(m,Int32(1));ccall(:srand48,Cvoid,(Clong,),123);a=Float16[];b=Float16[];oa=Int32[0];ob=Int32[0];oc=Int32[0];expected=Float32[];original_expected=Float32[]
    for g in 1:groups
        af=Float32[Float32(ccall(:drand48,Cdouble,())) for _ in 1:m[g]*k];bf=Float32[Float32(ccall(:drand48,Cdouble,())) for _ in 1:k*n];ag=Float16.(af);bg=Float16.(bf);append!(a,ag);append!(b,bg)
        for j in 0:n-1,i in 0:m[g]-1
            s=0f0;sf=0f0
            for q in 0:k-1
                s+=Float32(ag[i+q*m[g]+1])*Float32(bg[q+j*k+1]);sf+=af[i+q*m[g]+1]*bf[q+j*k+1]
            end
            push!(expected,Float32(Float16(s)));push!(original_expected,sf)
        end
        push!(oa,length(a));push!(ob,length(b));push!(oc,length(expected))
    end
    write(get(ENV,"GROUPGEMM_ORACLE_OUTPUT","groupgemm_rounded_expected.bin"),expected)
end
main()

using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function chemv_study(factory)
    n=370;repeats=1
    at=ComplexFloat[ComplexFloat(Float32(mod(7*col+11*row,61)-30)/2048f0,Float32(mod(3*col+5*row,59)-29)/2048f0) for col in 0:n-1 for row in 0:n-1]
    x=ComplexFloat[ComplexFloat(Float32(mod(i,31)-15)/128f0,Float32(mod(7*i,29)-14)/128f0) for i in 0:n-1]
    y=ComplexFloat[ComplexFloat(Float32(mod(i,19)-9)/256f0,Float32(mod(5*i,17)-8)/256f0) for i in 0:n-1]
    alpha=ComplexF64(3.14f0,1.59f0);beta=ComplexF64(2.71f0,8.28f0)
    expected=Float32[]
    for row in 0:n-1
        total=ComplexF64(0)
        for col in 0:n-1
            a=row>=col ? at[col*n+row+1] : at[row*n+col+1]
            value=row==col ? ComplexF64(a.Re,0) : row>col ? ComplexF64(a.Re,a.Im) : ComplexF64(a.Re,-a.Im)
            total+=value*ComplexF64(x[col+1].Re,x[col+1].Im)
        end
        value=alpha*total+beta*ComplexF64(y[row+1].Re,y[row+1].Im)
        push!(expected,Float32(real(value)),Float32(imag(value)))
    end
    da=CuArray(at);dx=CuArray(x);dy=CuArray(y);launch! = factory(da,dx,dy)
    readback=()->begin;h=Array(dy);Float32[v for z in h for v in (z.Re,z.Im)];end
    OwnedStudy.run_trials!(launch!,()->copyto!(dy,y),readback,expected,repeats;atol=1e-5,rtol=0.)
end

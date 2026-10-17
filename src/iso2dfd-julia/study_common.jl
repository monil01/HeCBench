using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function iso_study(launch!)
    nr=128;nc=160;steps=24;coef=(0.002f0*0.002f0)/(20f0*20f0)
    wave=Float32[0.016387336,-0.041464937,-0.067372555,0.386110067,0.812723635,0.416998396,0.076488599,-0.059434419,0.023680172,0.005611435,0.001823209,-0.000720549]
    initial=zeros(Float32,nr*nc);blank=similar(initial);fill!(blank,0f0);vel=fill(2250000f0,nr*nc)
    for s in 11:-1:0, row in nr÷2-s:nr÷2+s-1, col in nc÷2-s:nc÷2+s-1
        initial[row*nc+col+1]=wave[s+1]
    end
    old=copy(blank);cur=copy(initial)
    for step in 1:steps
        for row in 1:nr-2,col in 1:nc-2
            g=row*nc+col+1
            value=0f0;value+=cur[g+1]-2f0*cur[g]+cur[g-1]
            value+=cur[g+nc]-2f0*cur[g]+cur[g-nc]
            value*=coef*vel[g]
            old[g]=2f0*cur[g]-old[g]+value
        end
        old,cur=cur,old
    end
    expected=vcat(cur,old)
    a=CuArray(initial);bb=CuArray(blank);dv=CuArray(vel)
    current=Ref(a);previous=Ref(bb)
    function reset!()
        copyto!(a,initial);copyto!(bb,blank);current[]=a;previous[]=bb
    end
    function step!()
        launch!(previous[],current[],dv,coef,Int32(nr),Int32(nc))
        current[],previous[]=previous[],current[]
    end
    readback()=vcat(Array(current[]),Array(previous[]))
    OwnedStudy.run_trials!(step!,reset!,readback,expected,steps;atol=1e-4,rtol=1e-5)
end

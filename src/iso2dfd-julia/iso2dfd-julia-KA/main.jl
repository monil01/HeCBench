using CUDA,KernelAbstractions
@kernel function wave_step!(next,prev,vel,coef,nr,nc)
    col,row=@index(Global,NTuple)
    if row>1 && row<nr && col>1 && col<nc
        gid=(row-1)*nc+col
        @inbounds begin
            value=0f0;value+=prev[gid+1]-2f0*prev[gid]+prev[gid-1]
            value+=prev[gid+nc]-2f0*prev[gid]+prev[gid-nc]
            value*=coef*vel[gid]
            next[gid]=2f0*prev[gid]-next[gid]+value
        end
    end
end
function launch_ka!(next,prev,vel,coef,nr,nc)
    wave_step!(get_backend(next),(16,16))(next,prev,vel,coef,nr,nc;ndrange=(Int(nc),Int(nr)))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
iso_study(launch_ka!)

using CUDA,JACC
function wave_step!(col,row,next,prev,vel,coef,nr,nc)
    if row>1 && row<nr && col>1 && col<nc
        gid=(row-1)*nc+col
        @inbounds begin
            value=0f0;value+=prev[gid+1]-2f0*prev[gid]+prev[gid-1]
            value+=prev[gid+nc]-2f0*prev[gid]+prev[gid-nc]
            value*=coef*vel[gid]
            next[gid]=2f0*prev[gid]-next[gid]+value
        end
    end
    nothing
end
const BACKEND=JACC.get_backend(:cuda)
const SPEC=JACC.LaunchSpec{typeof(BACKEND)}(;threads=(16,16),blocks=(10,8),shmem_size=0,sync=false)
function launch_jacc!(next,prev,vel,coef,nr,nc)
    JACC.parallel_for(wave_step!,SPEC,(Int(nc),Int(nr)),next,prev,vel,coef,nr,nc)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
iso_study(launch_jacc!)

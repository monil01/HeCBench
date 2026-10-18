using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(k,a,b,o,idx)
 k==1&&return match1!(BACKEND,(128,1))(a,b,o,idx;ndrange=(128,1))
 k==2&&return match2!(BACKEND,(16,16))(a,b,o,idx;ndrange=(128,16))
 k==3&&return match3!(BACKEND,(16,16))(a,b,o,idx;ndrange=(128,16))
 k==4&&return match4!(BACKEND,(16,16))(a,b,o,idx;ndrange=(128,16))
 k==5&&return match5!(BACKEND,(16,16))(a,b,o,idx;ndrange=(128,16))
 k==6&&return match6!(BACKEND,(16,16))(a,b,o,idx;ndrange=(128,16))
 k==7&&return match7!(BACKEND,(32,8))(a,b,o,idx;ndrange=(128,8))
 k==8&&return match8!(BACKEND,(32,8))(a,b,o,idx;ndrange=(128,8))
 k==9&&return match9!(BACKEND,(32,4))(a,b,o,idx;ndrange=(128,4))
 k==10&&return match10!(BACKEND,(32,8))(a,b,o,idx;ndrange=(128,8))
end
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)

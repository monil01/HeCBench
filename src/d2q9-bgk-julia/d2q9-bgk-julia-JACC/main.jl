using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../device_common.jl")
allocate_model(a)=JACC.array(a)
# JACC.shared uses a single dynamic shared-memory base. Partition one 256-float
# array; source 0/1 integer scratch is represented exactly as float32 then
# converted back to Int32 in the unchanged serial count reduction.
const TEMPLATE=JACC.array(zeros(Float32,256))
const TEMPLATE_INPUT=Array(TEMPLATE)
@assert length(TEMPLATE_INPUT)==256 && all(iszero,TEMPLATE_INPUT)
println("STUDY_SHARED_TEMPLATE_VALUES ",repr(TEMPLATE_INPUT));flush(stdout)
const SPEC128=JACC.launch_spec(threads=128,blocks=8,shmem_size=1024,sync=false)
const SPEC256=JACC.launch_spec(threads=128,blocks=32,shmem_size=1024,sync=false)
function step_kernel!(i,a,b,ob,ps,pc,da,omega,nx,ny,t,template)
    id=Int32(i)-Int32(1);tx=id%128+1;group0=id÷128
    ii=id%nx;jj=id÷nx
    vals,v,c=step_values(a,ob,ii,jj,nx,ny,da,omega);store_fields!(b,ii+jj*nx,vals)
    tile=JACC.shared(template)
    @inbounds tile[tx]=v
    @inbounds tile[tx+128]=Float32(c)
    JACC.sync_workgroup()
    if tx==1
        s=0f0;n=Int32(0)
        @inbounds for q=1:128;s+=tile[q];n+=Int32(tile[q+128]);end
        idx=group0+t*(nx÷128)*ny+17
        @inbounds ps[idx]=s
        @inbounds pc[idx]=n
    end
    return nothing
end
function launch_step(a,b,ob,ps,pc,da,omega,nx,ny,t)
    spec=nx==128 && ny==8 ? SPEC128 : nx==256 && ny==16 ? SPEC256 : error("Undeclared geometry")
    JACC.parallel_for(spec,Int(nx)*Int(ny),step_kernel!,a,b,ob,ps,pc,da,omega,nx,ny,t,TEMPLATE)
    nothing
end
include("../study_common.jl")
main()

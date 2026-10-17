using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function geam_study(factory)
 T=!isempty(ARGS)&&ARGS[1]=="f64" ? Float64 : Float32
 nr=1025;nc=1537
 host=T[mod(row*7+col*11+div(row,17),13) for col in 0:nc-1,row in 0:nr-1]
 # Flatten row-major logical nr x nc: input index=row*nc+col.
 flat=T[mod(row*7+col*11+div(row,17),13) for row in 0:nr-1 for col in 0:nc-1]
 expected=vec(permutedims(reshape(flat,nc,nr)))
 input=reshape(CuArray(flat),nc,nr);output=CUDA.zeros(T,nr,nc);operation! = factory(input,output,nr,nc)
 println("STUDY_CONTRACT pure_matrix_transpose alpha1_beta0 nr1025_nc1537 full1575425entries type=",T," partial32tiles=true vendor_reference_vs_framework_custom_tile")
 OwnedStudy.run_trials!(operation!,()->fill!(output,zero(T)),()->vec(Array(output)),expected,10;atol=0.,rtol=0.)
end

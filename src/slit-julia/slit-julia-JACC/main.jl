using JACC,CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
const SLIT_BACKEND=JACC.get_backend(:cuda)
study_allocate(A)=JACC.array(SLIT_BACKEND,A)
# CUDA backend library interoperability; no JACC device FFT or cross-backend FFT API is claimed.
study_main(study_allocate)

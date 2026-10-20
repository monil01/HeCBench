# Integration entrypoint. Measured commands use the unchanged nested file.
length(ARGS)==2||error("Use make run for the locked full study, or pass compiled study_gpu.so and this benchmark parent under the shared GPU6 lock.")
include(joinpath(@__DIR__,"mcmd-julia-cuda.jl","main.jl"))

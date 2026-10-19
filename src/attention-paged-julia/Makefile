JULIA ?= julia
.PHONY: cuda-jl ka jacc
cuda-jl:
	$(JULIA) --project=../_julia_ka_env attention-paged-julia-cuda.jl/main.jl
ka:
	$(JULIA) --project=../_julia_ka_env attention-paged-julia-KA/main.jl
jacc:
	$(JULIA) --project=../_julia_jacc_env attention-paged-julia-JACC/main.jl

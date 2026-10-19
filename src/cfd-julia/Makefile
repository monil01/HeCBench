JULIA ?= julia
.PHONY: cuda-jl ka jacc
cuda-jl:
	$(JULIA) --project=../_julia_ka_env cfd-julia-cuda.jl/main.jl
ka:
	$(JULIA) --project=../_julia_ka_env cfd-julia-KA/main.jl
jacc:
	$(JULIA) --project=../_julia_jacc_env cfd-julia-JACC/main.jl

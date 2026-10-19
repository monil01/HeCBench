JULIA ?= julia
.PHONY: cuda-jl ka jacc
cuda-jl:
	$(JULIA) --project=../_julia_ka_env cmp-julia-cuda.jl/main.jl
ka:
	$(JULIA) --project=../_julia_ka_env cmp-julia-KA/main.jl
jacc:
	$(JULIA) --project=../_julia_jacc_env cmp-julia-JACC/main.jl

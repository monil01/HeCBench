JULIA ?= julia
.PHONY: cuda-jl ka jacc
cuda-jl:
	$(JULIA) --project=../_julia_ka_env bitcracker-julia-cuda.jl/main.jl
ka:
	$(JULIA) --project=../_julia_ka_env bitcracker-julia-KA/main.jl
jacc:
	$(JULIA) --project=../_julia_jacc_env bitcracker-julia-JACC/main.jl

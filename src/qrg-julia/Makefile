JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
.PHONY: main run study
main:
	@true
run:
	$(JULIA) $(PROJECT) main.jl 100000
study:
	$(JULIA) $(PROJECT) qrg-julia-cuda.jl/study.jl

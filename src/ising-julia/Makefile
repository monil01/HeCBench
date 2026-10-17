JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
.PHONY: main run study
main:
	@true
run:
	$(JULIA) $(PROJECT) main.jl -x 18 -y 34 -w 2 -n 17 -a 0.5
study:
	$(JULIA) $(PROJECT) ising-julia-cuda.jl/study.jl

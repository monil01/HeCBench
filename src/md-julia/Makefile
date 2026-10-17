JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=
.PHONY: main run study
main:
	@true
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 100
study:
	$(JULIA) $(PROJECT) md-julia-cuda.jl/study.jl

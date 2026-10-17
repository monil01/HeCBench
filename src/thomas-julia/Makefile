JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=
.PHONY: main run study clean
main:
	@true
run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 16384 64 100
study:
	$(JULIA) $(PROJECT) thomas-julia-cuda.jl/study.jl
clean:
	rm -f *.dat

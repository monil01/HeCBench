JULIA ?= julia
PROJECT = ../../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl 16 1
run-all:
	$(JULIA) --project=$(PROJECT) main.jl 16 1
	$(JULIA) --project=$(PROJECT) main.jl 16 2
	$(JULIA) --project=$(PROJECT) main.jl 16 4
	$(JULIA) --project=$(PROJECT) main.jl 16 8
	$(JULIA) --project=$(PROJECT) main.jl 32 1
	$(JULIA) --project=$(PROJECT) main.jl 32 2
	$(JULIA) --project=$(PROJECT) main.jl 32 4
	$(JULIA) --project=$(PROJECT) main.jl 32 8

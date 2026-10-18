JULIA ?= julia
PROJECT = ../../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl uint_pairs
run-all:
	$(JULIA) --project=$(PROJECT) main.jl float_keys
	$(JULIA) --project=$(PROJECT) main.jl float_pairs
	$(JULIA) --project=$(PROJECT) main.jl uint_keys
	$(JULIA) --project=$(PROJECT) main.jl uint_pairs

JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl
oracle:
	g++ -O2 -std=c++17 generate_oracle.cpp -o /tmp/hecbench_vmc_oracle
	/tmp/hecbench_vmc_oracle . > fixture_description.json
	python3 build_kinds.py

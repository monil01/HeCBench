JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl
fixture:
	g++ -O3 -mfma -ffp-contract=fast -std=c++17 generate_study.cpp -o /tmp/stsg_generate
	/tmp/stsg_generate

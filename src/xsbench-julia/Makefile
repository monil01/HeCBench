JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl
fixture:
	g++ -O2 -std=c++17 material_fixture.cpp -o /tmp/hecbench_xs_materials
	/tmp/hecbench_xs_materials materials.bin
	python3 generate_study_fixture.py

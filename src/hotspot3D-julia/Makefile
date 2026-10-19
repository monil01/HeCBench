JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl
fixture:
	g++ -O2 -std=c++17 generate_fixture.cpp -o /tmp/hotspot3D_fixture
	/tmp/hotspot3D_fixture

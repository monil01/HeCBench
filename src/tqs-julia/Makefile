JULIA ?= julia
PROJECT ?= ../_julia_ka_env
HOSTLIB = ../../.porting-state/porting_logs/tqs-julia-shared/study_host.so
run: host
	$(JULIA) --project=$(PROJECT) main.jl
host:
	mkdir -p $(dir $(HOSTLIB))
	g++ -O3 -std=c++17 -shared -fPIC -I/usr/local/cuda/include study_host.cpp -o $(HOSTLIB)
fixture:
	python3 generate_study_fixture.py

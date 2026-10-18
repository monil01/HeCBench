JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl

fixture:
	mkdir -p ../../.porting-state/porting_logs/pcc-julia-native
	g++ -O3 -std=c++17 -shared -fPIC -I../pcc-cuda normalization_native.cpp -o ../../.porting-state/porting_logs/pcc-julia-native/normalization_native.so
	python3 generate_study_fixture.py

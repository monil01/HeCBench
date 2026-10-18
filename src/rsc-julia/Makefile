JULIA ?= julia
PROJECT ?= ../_julia_ka_env
FIT_LIB = ../../.porting-state/porting_logs/rsc-julia-shared/study_fit.so
run:
	$(JULIA) --project=$(PROJECT) main.jl
fit:
	mkdir -p $(dir $(FIT_LIB))
	g++ -O3 -std=c++17 -fPIC -shared -pthread study_fit.cpp -o $(FIT_LIB)
fixture: fit
	python3 generate_study_fixture.py

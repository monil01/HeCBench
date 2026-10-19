JULIA ?= julia
PROJECT ?= ../_julia_ka_env
run:
	$(JULIA) --project=$(PROJECT) main.jl
fixture:
	g++ -O3 -std=c++17 -fopenmp generate_fixture.cpp -o /tmp/bn_fixture
	/tmp/bn_fixture
	python3 independent_score_oracle.py

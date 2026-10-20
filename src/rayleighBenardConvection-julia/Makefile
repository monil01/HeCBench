FIXTURES ?= /tmp/hecbench-stencil-ports-rbc-fixtures
CONFIG ?= /tmp/hecbench-rbc-reproduction-config.json
.PHONY: run run-all
run:
	julia main.jl $(FIXTURES)
run-all:
	python3 make_config.py --fixtures $(FIXTURES) --models cuda,cuda_jl,ka,jacc --output $(CONFIG)
	python3 run_study.py --config $(CONFIG)

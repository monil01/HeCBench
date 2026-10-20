FIXTURES ?= /tmp/hecbench-stencil-ports-slu-fixtures-v2
PRIVATE ?= /tmp/hecbench-stencil-ports-slu-private/manual_$(shell python3 -c 'import uuid;print(uuid.uuid4().hex)')
LIBRARY ?= /tmp/hecbench-stencil-ports-slu-study.so
.PHONY: all run run-all
all:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 build_program.py --mode shared --output "$(LIBRARY)"
run: all
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 julia --project=../_julia_ka_env main.jl "$(LIBRARY)" "$(FIXTURES)" "$(PRIVATE)"
run-all:
	python3 run_four_models.py --fixtures "$(FIXTURES)"

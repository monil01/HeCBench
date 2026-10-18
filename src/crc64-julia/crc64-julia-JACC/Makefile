JULIA ?= julia
run:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 JULIA_CUDA_MEMORY_POOL=none STUDY_OUTPUT_DIR=$${STUDY_OUTPUT_DIR:-/tmp} $(JULIA) --project=../../_julia_jacc_env main.jl

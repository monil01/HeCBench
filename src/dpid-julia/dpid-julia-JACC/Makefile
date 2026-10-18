JULIA ?= julia
run:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 STUDY_LAMBDA=$${STUDY_LAMBDA:-0.5} STUDY_OUTPUT_DIR=$${STUDY_OUTPUT_DIR:-/tmp} $(JULIA) --project=../../_julia_jacc_env main.jl

JULIA ?= julia
STUDY_OUTPUT_DIR ?= /tmp/aobench-JACC-artifacts
run:
	mkdir -p "$(STUDY_OUTPUT_DIR)"
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 STUDY_OUTPUT_DIR="$(STUDY_OUTPUT_DIR)" $(JULIA) --project=../../_julia_jacc_env main.jl

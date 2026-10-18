JULIA ?= julia
run:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib/libcublas.so.12.9.0.13 STUDY_OUTPUT_DIR=/tmp $(JULIA) --project=../../_julia_ka_env main.jl

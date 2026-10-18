JULIA ?= julia
run:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 $(JULIA) --project=../../_julia_ka_env main.jl

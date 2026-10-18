JULIA ?= julia
NVCC ?= nvcc
study_host.so: ../study_host.cu ../../atomicSystemWide-cuda/main.cu
	$(NVCC) -O3 -std=c++17 -arch=sm_70 -shared -Xcompiler=-fPIC ../study_host.cu -o $@
run: study_host.so
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 STUDY_HOST_LIBRARY=./study_host.so STUDY_OUTPUT_DIR=$${STUDY_OUTPUT_DIR:-/tmp} $(JULIA) --project=../../_julia_jacc_env main.jl

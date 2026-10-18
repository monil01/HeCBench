JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
.PHONY: main run clean
main:
	@echo "Both shared-atomic scatter kernels; ten exactdyadic study paths."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3
clean:
	@echo "Retain study fixtures."

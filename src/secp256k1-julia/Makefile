JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
REPEATS ?= 3
.PHONY: main run clean
main:
	@echo "Full512-point arithmetic GPU study; oldCPU/GPUcopy timing archived."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEATS)
clean:
	@echo "Study fixtures retained."

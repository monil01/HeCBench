JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
.PHONY: main run clean
main:
	@echo "Gated serialbase recurrence study; VLLM scan unverified."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1
clean:
	@echo "Retain study fixtures."

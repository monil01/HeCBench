JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
REPEATS ?= 20
.PHONY: main run clean
main:
	@echo "FP16/BF16 scaledmasked softmax, named16paths; conditionalnativecompilepatch."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEATS)
clean:
	@echo "Retain study fixtures."

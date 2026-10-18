JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
REPEATS ?= 3
.PHONY: main run clean
main:
	@echo "Named six-path convolution study; original files and CLI archived in reference_history."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEATS)
clean:
	@echo "Study fixtures retained."

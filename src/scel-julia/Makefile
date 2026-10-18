JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?= flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6
REPEATS ?= 20
.PHONY: main run clean
main:
	@echo "Named fifteen-path loss study; original dummy archived."
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEATS)
clean:
	@echo "Study fixtures retained."

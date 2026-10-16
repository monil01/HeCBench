JULIA ?= julia
PROJECT ?= --project=../../_julia_jacc_env
LAUNCHER ?=
ARGS ?= 256 256 100 25 5
main:
	@true
run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
clean:
	@true

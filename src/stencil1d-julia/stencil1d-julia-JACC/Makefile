JULIA ?= julia
PROJECT ?= --project=../../_julia_jacc_env
LAUNCHER ?=
ARGS ?= 1048576 100

main:
	@true

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)

clean:
	@true

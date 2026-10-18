JULIA ?= julia
PROJECT ?= ../_julia_ka_env
LAUNCHER ?=
.PHONY: main run clean
main:
	@true
run: main
	$(LAUNCHER) $(JULIA) --project=$(PROJECT) --startup-file=no main.jl
clean:
	@true

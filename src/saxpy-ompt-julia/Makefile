JULIA ?= julia
PROJECT ?= --project=$(abspath ../_julia_env)
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4194304 100

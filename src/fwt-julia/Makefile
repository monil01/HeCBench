JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=
ARGS ?= 2 20 7

main:
	@true

clean:
	rm -f

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
